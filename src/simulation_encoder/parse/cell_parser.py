import dataclasses
from typing import Tuple, Optional, Any

import numpy as np
from data_classes import CellMetrics

import pandas as pd


class CellParser:
    def __init__(self):
        self.data_dir = "../../data/ARCADE"
        self.timepoints = [(x / 2.0) for x in range(0, 31)]

    def parse_cell_metrics(self, key):
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

        file_df.to_csv(f"{self.data_dir}/{key}cell_metrics.csv", index=False)

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
