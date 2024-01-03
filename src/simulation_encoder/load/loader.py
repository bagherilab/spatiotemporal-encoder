import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, Subset

from sklearn.model_selection import train_test_split


class ArcadeDataset(Dataset):
    def __init__(
        self,
        keys: list[str],
        split_ratio: float = 0.2,
        cell: bool = True,
        graph: bool = True,
        seed: int = 42,
    ):
        self.data_dir = "../../data/ARCADE"
        self.timepoints = [(x / 2.0) for x in range(0, 31)]
        self.cell_data = None
        self.graph_data = None
        if cell:
            self.cell_dfs = [pd.read_csv(f"{self.data_dir}/{key}cell_metrics.csv") for key in keys]
        if graph:
            self.graph_dfs = [
                pd.read_csv(f"{self.data_dir}/{key}graph_metrics.csv") for key in keys
            ]

        self._align_data()

        self.train_indices, self.test_indices = self._split_data(split_ratio, seed)

    def train_subset(self):
        return Subset(self, self.train_indices)

    def test_subset(self):
        return Subset(self, self.test_indices)

    def _align_data(self):
        aligned_dfs = [
            pd.merge(
                cell_df,
                graph_df,
                on=["seed", "timepoint", "name", "context", "layout"],
                how="inner",
            )
            for cell_df, graph_df in zip(self.cell_dfs, self.graph_dfs)
        ]
        self.data = pd.concat(aligned_dfs, ignore_index=True)
        self.data["sim_key"] = self.data["name"].astype(str) + self.data["seed"].astype(str)
        self.features = self.data.drop(
            columns=["seed", "timepoint", "name", "context", "layout", "sim_key"]
        )
        self.features = self.features.fillna(0)
        self.features = self.features.apply(pd.to_numeric)
        self.features = torch.tensor(self.features.values)

    def _split_data(self, split_ratio, random_state):
        # Extract unique seeds and names
        unique_simulations = self.data["sim_key"].unique()

        train_sims, test_sims = train_test_split(
            unique_simulations, test_size=split_ratio, random_state=random_state
        )

        train_indices = self.data[self.data["sim_key"].isin(train_sims)].index
        test_indices = self.data[self.data["sim_key"].isin(test_sims)].index

        return train_indices, test_indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        labels = {
            "seed": self.data["seed"][idx],
            "timepoint": self.data["timepoint"][idx],
            "name": self.data["name"][idx],
        }
        return self.features[idx], labels
