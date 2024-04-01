import os
import dataclasses
from typing import Tuple, Optional, Any

import networkx as nx
import igraph as ig

import numpy as np
from data_classes import GraphMetrics

from collections import OrderedDict

from karateclub import Graph2Vec

from utils.s3_utils import download_file, upload_file

import pandas as pd


class GraphEmbedder:
    def __init__(self):
        self.data_dir = "../../../data/ARCADE"
        self.timepoints = [(x / 2.0) for x in range(2, 31)]
        self.object_prefix = "jevarts/encoder/parsed_sims"
        self.bucket = "bagherilab-working"

    def embed_graph(self, key) -> pd.DataFrame:
        download_file(self.bucket, f"{self.object_prefix}/raw/{key}graph.csv", f"{self.data_dir}/{key}graph.csv")
        graph_df = pd.read_csv(f"{self.data_dir}/{key}graph.csv")
        seeds = sorted(graph_df.seed.unique())

        graphs = []
        metadata_list = []

        for seed in seeds:
            print("seed: ", seed)
            for timepoint in self.timepoints:
                simulation_df = graph_df.loc[
                    (graph_df["seed"] == seed) & (graph_df["time"] == timepoint)
                ]

                edges = self.get_edges(simulation_df)
                graph = self.make_networkx_graph(edges)
                graphs.append(graph)

                metadata = {"seed": seed, "timepoint": timepoint, "name": key}
                metadata_list.append(metadata)

        graph2vec = Graph2Vec(dimensions=2)
        graph2vec.fit(graphs)
        graph_embedding = graph2vec.get_embedding()

        metadata_df = pd.DataFrame(metadata_list)
        embeddings_df = pd.DataFrame(graph_embedding)

        graph_embedding_df = pd.concat([embeddings_df, metadata_df], axis=1)
        graph_embedding_df.to_csv(f"{self.data_dir}/{key}graph_embedding.csv", index=False)
        # upload_file(f"{self.data_dir}/{key}graph_embedding.csv", self.bucket, f"{self.object_prefix}/embeddings/{key}graph_embedding.csv")
        os.remove(f"{self.data_dir}/{key}graph.csv")
                
        return graph_embedding_df


    def make_networkx_graph(self, edges: list[Tuple[Any, Any]]) -> nx.DiGraph:
        """
        Creates a networkx graph from provided edges for certain graph metric calculations and returns
        a directed version of the graph

        Parameters
        ----------
        edges :
            List of edges defined by end nodes (i.e. [[1,2], [2,4]...]) of the graph to be analyzed
        weights :
            (Optional) List of weights in the same order as the list of edges

        Returns
        -------
        dir_graph
            Directed graph
        """

        nodes = []
        for edge in edges:
            nodes.extend(edge)
        nodes = list(OrderedDict.fromkeys(nodes))

        node_mapping = {node: index for index, node in enumerate(nodes)}

        nx_graph = nx.DiGraph()
        nx_graph.add_nodes_from(range(len(nodes)))

        for edge in edges:
            nx_graph.add_edge(node_mapping[edge[0]], node_mapping[edge[1]])

        return nx_graph

    def get_edges(self, graph_df: pd.DataFrame) -> list[Tuple[Any, Any]]:
        """
        Creates a list of edges from a dataframe of edges

        Parameters
        ----------
        graph_df :
            Dataframe of edges

        Returns
        -------
        edges
            List of edges
        """
        node_ax_indeces = list(graph_df.loc[:, "fromx"])
        node_ay_indeces = list(graph_df.loc[:, "fromy"])
        node_a_indeces = list(zip(node_ax_indeces, node_ay_indeces))

        node_bx_indeces = list(graph_df.loc[:, "tox"])
        node_by_indeces = list(graph_df.loc[:, "toy"])
        node_b_indeces = list(zip(node_bx_indeces, node_by_indeces))

        edges = list(zip(node_a_indeces, node_b_indeces))
        return edges

class GraphParser:
    def __init__(self):
        self.data_dir = "../../data/ARCADE"
        self.timepoints = [(x / 2.0) for x in range(0, 31)]
        self.object_prefix = "jevarts/encoder/parsed_sims"
        self.bucket = "bagherilab-working"

    def parse_graph_metrics(self, key):
        download_file(self.bucket, f"{self.object_prefix}/raw/{key}graph.csv", f"{self.data_dir}/{key}graph.csv")

        graph_df = pd.read_csv(f"{self.data_dir}/{key}graph.csv")
        seeds = sorted(graph_df.seed.unique())

        file_df = pd.DataFrame()

        for seed in seeds:
            print("seed: ", seed)
            for timepoint in self.timepoints:
                # Get sub df for seed and timepoint
                simulation_df = graph_df.loc[
                    (graph_df["seed"] == seed) & (graph_df["time"] == timepoint)
                ]
                node_ax_indeces = list(simulation_df.loc[:, "fromx"])
                node_ay_indeces = list(simulation_df.loc[:, "fromy"])
                node_a_indeces = list(zip(node_ax_indeces, node_ay_indeces))

                node_bx_indeces = list(simulation_df.loc[:, "tox"])
                node_by_indeces = list(simulation_df.loc[:, "toy"])
                node_b_indeces = list(zip(node_bx_indeces, node_by_indeces))

                edges = list(zip(node_a_indeces, node_b_indeces))
                metrics = self.igraph_graph_metrics(edges)
                metrics.seed = seed
                metrics.timepoint = timepoint
                metrics.name = key
                metrics.context = self._get_context(key)
                metrics.layout = self._get_layout(key)

                network_metrics_dict = dataclasses.asdict(metrics)
                network_metrics_df = pd.DataFrame.from_dict([network_metrics_dict])

                file_df = pd.concat([file_df, network_metrics_df], ignore_index=True)

        os.remove(f"{self.data_dir}/{key}graph.csv")
        file_df.to_csv(f"{self.data_dir}/{key}graph_metrics.csv", index=False)
        upload_file(f"{self.data_dir}/{key}graph_metrics.csv", self.bucket, f"{self.object_prefix}/metrics/{key}graph_metrics.csv")

    def _get_layout(self, key: str) -> str:
        name_chunks = key.split("_")
        return name_chunks[1]

    def _get_context(self, key: str) -> str:
        name_chunks = key.split("_")
        return name_chunks[0]

    def igraph_graph_metrics(
        self, edges: list[Tuple[Any, Any]], weights: Optional[list[float]] = None
    ) -> GraphMetrics:
        """
        Calculate graph metrics from a set of edges

        Parameters
        ----------
        edges :
            List of edges defined by end nodes (i.e. [[1,2], [2,4]...]) of the graph to be analyzed
        weights :
            (Optional) List of weights in the same order as the list of edges

        Returns
        -------
        metrics
            GraphMetrics object that holds graph metrics
        """

        if len(edges) == 0:
            raise ValueError("Passed edges do not create valid graph.")
        if not weights:
            weights = [1.0] * len(edges)

        igraph = self._make_igraph(edges, weights)
        # node_inverse_distances_from_center = [
        #     self._inverse_distance_from_center(node["name"]) for node in igraph.vs
        # ]

        connected: bool = igraph.is_connected("weak")

        m_dict = {}

        m_dict["nodes"] = igraph.vcount()
        m_dict["edges"] = igraph.ecount()

        degree_metrics_dict = self._calc_degree_metrics(igraph)
        m_dict.update(degree_metrics_dict)

        distance_metrics_dict = self._calc_distance_metrics(
            igraph, connected
        )
        m_dict.update(distance_metrics_dict)

        betweeness_metrics_dict = self._calc_betweenness_metric(
            igraph, connected
        )
        m_dict.update(betweeness_metrics_dict)

        coreness_metrics_dict = self._calc_coreness_metric(
            igraph, connected
        )
        m_dict.update(coreness_metrics_dict)

        m_dict["avg_clustering"] = self._calc_clustering_metric(igraph)
        m_dict["components"] = self._calc_n_components(igraph, connected)

        metrics = GraphMetrics(**m_dict)
        return metrics

    def _calc_distance_metrics(
        self, graph: ig.Graph, connected: bool, avg_weight: Optional[list[float]] = None
    ) -> dict[str, float]:
        """Helper function to calculatedistance based metrics from igraph"""
        dist_m_dict = {}
        if not connected:
            return {
                "radius": float("inf"),
                "diameter": float("inf"),
                "avg_eccentricity": float("inf"),
                "avg_closeness": float("inf"),
                "avg_shortest_path": float("inf"),
                "avg_eccentricity_weighted": float("inf"),
                "avg_closeness_weighted": float("inf"),
            }

        n_nodes = graph.vcount()
        path_norm_factor = n_nodes * (n_nodes - 1)

        eccs: list[float] = []
        closeness: list[float] = []
        total_distance: int = 0
        try:
            for node in graph.vs:
                distance_vector = graph.distances(source=node, mode="all", weights="weight")[0]
                distance_vector = np.array(list(distance_vector))
                reachable = np.count_nonzero(~np.isinf(distance_vector))
                distances = np.nan_to_num(distance_vector, posinf=0.0)
                if reachable <= 1:
                    continue
                distance = np.sum(distances)
                eccs.append(max(distances))
                closeness.append((reachable - 1) * (reachable - 1) / np.sum(distance) / (n_nodes))
                total_distance += distance
        except ig._igraph.InternalError:
            return {
                "radius": float("inf"),
                "diameter": float("inf"),
                "avg_eccentricity": float("inf"),
                "avg_closeness": float("inf"),
                "avg_shortest_path": float("inf"),
                "avg_eccentricity_weighted": float("inf"),
                "avg_closeness_weighted": float("inf"),
            }

        dist_m_dict["radius"] = min(eccs)
        dist_m_dict["diameter"] = max(eccs)
        dist_m_dict["avg_eccentricity"] = float(np.mean(eccs))
        dist_m_dict["avg_closeness"] = float(np.mean(np.nan_to_num(closeness, posinf=0.0)))
        dist_m_dict["avg_shortest_path"] = float(total_distance / path_norm_factor)
        if avg_weight:
            dist_m_dict["avg_eccentricity_weighted"] = self._weighted_average(eccs, avg_weight)
            closeness_list = np.nan_to_num(closeness, posinf=0.0).tolist()
            dist_m_dict["avg_closeness_weighted"] = self._weighted_average(
                closeness_list, avg_weight
            )
        else:
            dist_m_dict["avg_eccentricity_weighted"] = float("inf")
            dist_m_dict["avg_closeness_weighted"] = float("inf")

        return dist_m_dict

    def _calc_betweenness_metric(
        self, graph: ig.Graph, connected: bool, avg_weight: Optional[list[float]] = None
    ) -> dict[str, float]:
        """Helper function to calcuate normalized betweenness from igraph"""
        between_m_dict = {}
        if not connected:
            return {
                "avg_betweenness": float("inf"),
                "avg_betweenness_weighted": float("inf"),
            }

        n_nodes = graph.vcount()
        betweenness_norm_factor = (n_nodes - 1) * (n_nodes - 2)
        try:
            betweenness = np.array(graph.betweenness(weights="weight")) / betweenness_norm_factor
            between_m_dict["avg_betweenness"] = np.mean(betweenness)
            if avg_weight:
                between_m_dict["avg_betweenness_weighted"] = self._weighted_average(
                    betweenness, avg_weight
                )
            else:
                between_m_dict["avg_betweenness_weighted"] = float("inf")

            return between_m_dict
        except ig._igraph.InternalError:
            return {
                "avg_betweenness": float("inf"),
                "avg_betweenness_weighted": float("inf"),
            }

    def _calc_degree_metrics(
        self, graph: ig.Graph, avg_weight: Optional[list[float]] = None
    ) -> dict[str, float]:
        """
        Helper function to calculate the average degree (number of edges per node) from igraph

        Parameters
        ----------
        graph :
            igraph Graph

        Returns
        -------
        return_tuple
            tuple containing (average_indegree, average_outdegree, average_degree)
        """
        degree_m_dict = {}
        modes = ("in", "out", "all")
        (
            degree_m_dict["avg_in_degrees"],
            degree_m_dict["avg_out_degrees"],
            degree_m_dict["avg_degree"],
        ) = tuple(np.mean(graph.degree(mode=mode)) for mode in modes)
        if avg_weight:
            (
                degree_m_dict["avg_in_degrees_weighted"],
                degree_m_dict["avg_out_degrees_weighted"],
                degree_m_dict["avg_degree_weighted"],
            ) = tuple(self._weighted_average(graph.degree(mode=mode), avg_weight) for mode in modes)
        else:
            (
                degree_m_dict["avg_in_degrees_weighted"],
                degree_m_dict["avg_out_degrees_weighted"],
                degree_m_dict["avg_degree_weighted"],
            ) = tuple(float("inf") for _ in modes)
        return degree_m_dict

    def _calc_clustering_metric(self, graph: ig.Graph) -> float:
        """
        Helper function to calculate the global transitivity,
        i.e. the Clustering coefficient from igraph
        """
        return graph.transitivity_avglocal_undirected()

    def _calc_coreness_metric(
        self, graph: ig.Graph, connected: bool, avg_weight: Optional[list[float]] = None
    ) -> dict[str, float]:
        """Helper function to get the assortivity from igraph."""
        coreness_m_dict = {}
        if not connected:
            return {"avg_coreness": float("inf"), "avg_coreness_weighted": float("inf")}

        coreness = graph.coreness(mode="all")
        coreness_m_dict["avg_coreness"] = float(np.average(coreness))
        if avg_weight:
            coreness_m_dict["avg_coreness_weighted"] = self._weighted_average(coreness, avg_weight)
        else:
            coreness_m_dict["avg_coreness_weighted"] = float("inf")
        return coreness_m_dict

    def _calc_n_components(self, graph: ig.Graph, connected: bool) -> int:
        """Helper function to get the number of components (subgraphs) from igraph."""
        if connected:
            return 1

        return len(graph.decompose(mode="weak"))

    def _make_igraph(self, edges: list[Tuple[Any, Any]], weights: list[float]) -> ig.Graph:
        """
        Creates an igraph graph from provided edges for certain graph metric calculations and returns
        a directed version of the graph

        Parameters
        ----------
        edges :
            List of edges defined by end nodes (i.e. [[1,2], [2,4]...]) of the graph to be analyzed
        weights :
            (Optional) List of weights in the same order as the list of edges

        Returns
        -------
        dir_graph
            Directed igraph
        """
        dir_graph = ig.Graph.TupleList(edges=edges, directed=True)
        dir_graph.es["weight"] = weights
        return dir_graph

    def _inverse_distance_from_center(self, node: Tuple[Any, Any]) -> float:
        """Helper function to calculate the distance from the center of the graph"""

        return 1 / np.sqrt(
            (int(node[0]) - self.center[0]) ** 2 + (int(node[1]) - self.center[1]) ** 2
        )

    def _weighted_average(self, values: list[float], weights: list[float]) -> float:
        # print(weights)
        total_weight = sum(weights)
        weighted_sum = sum(v * w for v, w in zip(values, weights))
        return float(weighted_sum / total_weight)
