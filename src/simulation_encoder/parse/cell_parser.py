import os
import dataclasses
from typing import Tuple, Optional, Any

import numpy as np
from data_classes import CellMetrics

import pandas as pd

from utils.s3_utils import download_file, upload_file


class CellEmbedder:
    def __init__(self):
        self.data_dir = "../../data/ARCADE"
        self.timepoints = [(x / 2.0) for x in range(0, 31)]
        self.object_prefix = "jevarts/encoder/parsed_sims"
        self.bucket = "bagherilab-working"

    def embed_cells(self, key) -> pd.DataFrame:
        # download_file(self.bucket, f"{self.object_prefix}/raw/{key}cell.csv", f"{self.data_dir}/{key}cell.csv")
        cell_df = pd.read_csv(f"{self.data_dir}/{key}cell.csv")
        seeds = sorted(cell_df.seed.unique())

        metadata_list = []

        for seed in seeds:
            print("seed: ", seed)
            for timepoint in self.timepoints:
                simulation_df = cell_df[(cell_df.seed == seed) & (cell_df.timepoint == timepoint)]
                return simulation_df
                

        
        # file_df.to_csv(f"{self.data_dir}/{key}cell_metrics.csv", index=False)
        # upload_file(f"{self.data_dir}/{key}cell_metrics.csv", self.bucket, f"{self.object_prefix}/metrics/{key}cell_metrics.csv")
        os.remove(f"{self.data_dir}/{key}cell.csv")



class CellParser:
    def __init__(self):
        self.data_dir = "../../data/ARCADE"
        self.timepoints = [(x / 2.0) for x in range(0, 31)]
        self.object_prefix = "jevarts/encoder/parsed_sims"
        self.bucket = "bagherilab-working"

    def parse_cell_metrics(self, key):
        download_file(self.bucket, f"{self.object_prefix}/raw/{key}cell.csv", f"{self.data_dir}/{key}cell.csv")

        cell_df = pd.read_csv(f"{self.data_dir}/{key}cell.csv")
        seeds = sorted(cell_df.seed.unique())

        file_df = pd.DataFrame()

        for seed in seeds:
            print("seed: ", seed)
            for timepoint in self.timepoints:
                simulation_df = cell_df[(cell_df.seed == seed) & (cell_df.timepoint == timepoint)]

                metrics = self.cell_metrics(simulation_df)
                metrics.seed = seed
                metrics.timepoint = timepoint
                metrics.name = key
                metrics.context = self._get_context(key)
                metrics.layout = self._get_layout(key)

                cell_metrics_dict = dataclasses.asdict(metrics)
                cell_metrics_df = pd.DataFrame.from_dict([cell_metrics_dict])

                file_df = pd.concat([file_df, cell_metrics_df], ignore_index=True)

        os.remove(f"{self.data_dir}/{key}cell.csv")
        file_df.to_csv(f"{self.data_dir}/{key}cell_metrics.csv", index=False)
        upload_file(f"{self.data_dir}/{key}cell_metrics.csv", self.bucket, f"{self.object_prefix}/metrics/{key}cell_metrics.csv")

    def _get_layout(self, key: str) -> str:
        name_chunks = key.split("_")
        return name_chunks[1]

    def _get_context(self, key: str) -> str:
        name_chunks = key.split("_")
        return name_chunks[0]

    def cell_metrics(self, simulation_df: pd.DataFrame) -> CellMetrics:
        num_cells = len(simulation_df)
        avg_volume = np.mean(simulation_df.volume)
        percent_undecided = np.mean(simulation_df.state == 0)
        percent_apoptotic = np.mean(simulation_df.state == 1)
        percent_quiescent = np.mean(simulation_df.state == 2)
        percent_migrating = np.mean(simulation_df.state == 3)
        percent_proliferating = np.mean(simulation_df.state == 4)
        percent_senescent = np.mean(simulation_df.state == 5)
        percent_necrotic = np.mean(simulation_df.state == 6)
        cancer_fraction = np.mean(simulation_df.population == 0)

        return CellMetrics(
            num_cells=num_cells,
            avg_volume=avg_volume,
            percent_undecided=percent_undecided,
            percent_apoptotic=percent_apoptotic,
            percent_quiescent=percent_quiescent,
            percent_migrating=percent_migrating,
            percent_proliferating=percent_proliferating,
            percent_senescent=percent_senescent,
            percent_necrotic=percent_necrotic,
            cancer_fraction=cancer_fraction,
        )
