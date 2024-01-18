import os
import json
import ntpath

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.s3_utils import download_file, list_s3_files, upload_file
from parse.graph_parser import GraphParser
from parse.cell_parser import CellParser




class Parser(ABC):
    @abstractmethod
    def parse_files_to_csv(self, key) -> None:
        pass


class ArcadeParser:
    def __init__(self):
        self.timepoints = [(x / 2.0) for x in range(0, 31)]
        self.data_dir = "../../data/ARCADE"
        self.bucket = "bagherilab-working"
        self.load_prefix = "jcain/emulation_sims/outputs"
        self.save_prefix = "jevarts/encoder/parsed_sims"

    def parse_files_to_csv(self, keys: list[str]) -> None:
        for key in keys:
            cell_files = list_s3_files(
                self.bucket, self.load_prefix, include=[key], exclude=[".GRAPH"]
            )
            file_pd = pd.DataFrame()
            for file_name in cell_files:

                local_path = f"{self.data_dir}/{file_name}"
                s3_path = f"{self.load_prefix}/{file_name}"
                download_file(self.bucket, s3_path, local_path)
                parsed_df = self._parse_cells(local_path)
                file_pd = pd.concat([file_pd, parsed_df], ignore_index=True)
                os.remove(local_path)

            file_pd.to_csv(f"{self.data_dir}/{key}cell.csv", index=False)
            upload_file(f"{self.data_dir}/{key}cell.csv", self.bucket, f"{self.save_prefix}/raw/{key}cell.csv")
            os.remove(f"{self.data_dir}/{key}cell.csv")

            graph_files = list_s3_files(self.bucket, self.load_prefix, include=[key, ".GRAPH"])
            file_pd = pd.DataFrame()
            for file_name in graph_files:
                local_path = f"{self.data_dir}/{file_name}"
                s3_path = f"{self.load_prefix}/{file_name}"
                download_file(self.bucket, s3_path, local_path)
                parsed_df = self._parse_graph(local_path)
                file_pd = pd.concat([file_pd, parsed_df], ignore_index=True)
                os.remove(local_path)

            file_pd.to_csv(f"{self.data_dir}/{key}graph.csv", index=False)
            upload_file(f"{self.data_dir}/{key}graph.csv", self.bucket, f"{self.save_prefix}/raw/{key}graph.csv")
            os.remove(f"{self.data_dir}/{key}graph.csv")

    def parse_graph_metrics_to_csv(self, keys: list[str]) -> None:
        graph_parser = GraphParser()
        for key in keys:
            graph_parser.parse_graph_metrics(key)

    def parse_cell_metrics_to_csv(self, keys: list[str]) -> None:
        cell_parser = CellParser()
        for key in keys:
            cell_parser.parse_cell_metrics(key)

    def _parse_cells(self, sim_file) -> pd.DataFrame:
        parsed_data = []
        parsed_df = pd.DataFrame()

        with open(sim_file, "r") as f:
            sim_json = json.load(f)

        for time_index, timepoint in enumerate(self.timepoints):
            sim_timepoint = sim_json["timepoints"][time_index]["cells"]

            for location, cells in sim_timepoint:
                u, v, w, z = map(int, location[:4])

                for cell in cells:
                    population, state, position, volume, cell_cycle = cell[1:6]
                    cycle = np.round(np.mean(cell_cycle)) if cell_cycle else 0
                    seed = self._get_seed(sim_file)

                    data_list = [
                        timepoint,
                        u,
                        v,
                        w,
                        z,
                        position,
                        int(population),
                        int(state),
                        np.round(volume),
                        cycle,
                        seed,
                    ]

                    parsed_data.append(data_list)

        columns = [
            "timepoint",
            "u",
            "v",
            "w",
            "z",
            "position",
            "population",
            "state",
            "volume",
            "cycle",
            "seed",
        ]
        if parsed_data:
            parsed_df = pd.DataFrame(parsed_data, columns=columns)

        return parsed_df

    def _parse_graph(self, graph_file) -> pd.DataFrame:
        parsed_data = []
        parsed_df = pd.DataFrame()

        with open(graph_file, "r") as f:
            graph_json = json.load(f)

        seed = graph_json["seed"]
        csv_data = [
            "seed",
            "time",
            "fromx",
            "fromy",
            "fromz",
            "frompressure",
            "fromoxygen",
            "tox",
            "toy",
            "toz",
            "topressure",
            "tooxygen",
            "CODE",
            "RADIUS",
            "LENGTH",
            "WALL",
            "SHEAR",
            "CIRCUM",
            "FLOW",
        ]

        for timepoint in graph_json["timepoints"]:
            time = timepoint["time"]

            for edge in timepoint["graph"]:
                fromx, fromy, fromz, frompressure, fromoxygen = edge[0][:5]
                tox, toy, toz, topressure, tooxygen = edge[1][:5]
                code, radius, length, wall, shear, circum, flow = edge[2][:7]

                data_list = [
                    seed,
                    time,
                    fromx,
                    fromy,
                    fromz,
                    frompressure,
                    fromoxygen,
                    tox,
                    toy,
                    toz,
                    topressure,
                    tooxygen,
                    code,
                    radius,
                    length,
                    wall,
                    shear,
                    circum,
                    flow,
                ]

                parsed_data.append(data_list)

        parsed_df = pd.DataFrame(parsed_data, columns=csv_data)

        return parsed_df

    def _get_seed(self, f) -> None:
        f_path = ntpath.dirname(f)
        base = ntpath.basename(f)
        remove_extension = os.path.splitext(base)[0]
        remove_suffix = os.path.splitext(remove_extension)[0]
        seed = int(remove_suffix[-2:])
        extension = os.path.splitext(base)[1]
        key = remove_suffix[:-3]

        return seed