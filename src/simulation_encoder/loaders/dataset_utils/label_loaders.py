import os
import json
from typing import Optional

import pandas as pd

from torch.utils.data import Dataset


class CSVLoader(Dataset):
    """
    Loader class for loading labeled data from a CSV file.

    Parameters
    ----------
    exp_id : str
        Experiment ID.
    labels : list[str]
        List of target labels.
    """

    def __init__(
        self, conf_name: str, exp_id: str, model: str, dataset_name: str, labels: list[str]
    ) -> None:
        self.exp_id = exp_id
        self.dataset_name = dataset_name
        self.data_path = f"results/{conf_name}/{exp_id}/{model}/{dataset_name}"
        self.labels = labels
        self._load_data()

    def __len__(self) -> int:
        return len(self._X_train) + len(self._X_val) + len(self._X_test)

    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("This method is not implemented yet. Use get_dataloader instead.")

    def get_data(
        self,
        dataset_type: str,
        feature_timepoint: Optional[int] = None,
        response_timepoint: Optional[int] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if dataset_type == "train":
            X, y = self._X_train, self._y_train
        elif dataset_type == "val":
            X, y = self._X_val, self._y_val
        elif dataset_type == "test":
            X, y = self._X_test, self._y_test
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

        if feature_timepoint:
            X = X[X["timepoint"] == feature_timepoint]

        if response_timepoint:
            y = y[y["timepoint"] == response_timepoint]

        # Align based on seed_key
        if feature_timepoint is not None or response_timepoint is not None:
            common_seed_keys = set(X["seed_key"]).intersection(y["seed_key"])
            X = X[X["seed_key"].isin(common_seed_keys)]
            y = y[y["seed_key"].isin(common_seed_keys)]

            # Sort by seed_key to ensure alignment
            X = X.sort_values("seed_key").reset_index(drop=True)
            y = y.sort_values("seed_key").reset_index(drop=True)

        return X.drop(["timepoint", "seed_key"], axis=1), y.drop(["timepoint", "seed_key"], axis=1)

    def _load_data(self) -> None:
        self._X_train, self._y_train = self._load_csv("train")
        self._X_val, self._y_val = self._load_csv("val")
        self._X_test, self._y_test = self._load_csv("test")

    def _load_csv(self, dataset_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        file_path = os.path.join(self.data_path, f"encoded_data/{dataset_type}.csv")
        data = pd.read_csv(file_path)
        self.feature_cols = [col for col in data.columns if col.startswith("dim_")]
        X, y = (
            data.loc[:, self.feature_cols + ["timepoint", "seed_key"]],
            data.loc[:, self.labels + ["timepoint", "seed_key"]],
        )
        return X, y


class LabelLoader:
    """
    Loads labels (activity, growth, and symmetry) to match with corresponding images.

    Parameters
    ----------
    label_dir : str
        Path to the directory containing the labels.

    """

    def __init__(self, label_dir: str):
        self.label_dir = label_dir

    def get_labels(self, label: str, key: str, time: float, seed: int) -> float:
        file = f"{self.label_dir}/VASCULAR_FUNCTION_{key}.SEEDS.{label}.json"

        with open(file, "r", encoding="utf-8") as f:
            vals = json.load(f)
        for val in vals:
            if val["time"] == time:
                return val["_"][seed]

        return float("nan")
