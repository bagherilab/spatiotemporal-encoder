import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    """Dataset for time-series or otherwise ordered data"""

    def __init__(self, sequences: dict):
        self.sequences = sequences
        self.sample_ids = list(sequences.keys())

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> tuple[torch.tensor, int]:
        sample_id = self.sample_ids[idx]
        sequence = torch.tensor(self.sequences[sample_id], dtype=torch.float32)
        return sequence, sample_id


class SequenceLoader:
    """TODO"""

    def __init__(
        self,
        data_path: str,
        sequence_col: str = "timepoint",
        id_col: str = "sample_id",
        val_split: float = 0.2,
        test_split: float = 0.2,
        batch_size: int = 16,
        random_seed: int = 42,
    ):
        self.data = self._load_csv(data_path)
        self.sequence_col = sequence_col
        self.id_col = id_col
        self.feature_cols = [
            col
            for col in self.data.columns
            if col not in [self.sequence_col, self.id_col]
        ]
        self.num_dims = len(self.feature_cols)

        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.random_seed = random_seed

        self.sequences = {}
        for sample_id, group in self.data.groupby(self.id_col):
            sorted_group = group.sort_values(by=self.sequence_col)
            self.sequences[sample_id] = sorted_group[self.feature_cols].values

        self.sequence_len = len(group)
        self._train_ids, self._val_ids, self._test_ids = self._split_data()

    def get_dataloader(self, dataset_type: str) -> DataLoader:
        """Returns DataLoader for the specified dataset type (train, val, test)"""
        if dataset_type == "train":
            ids = self._train_ids
        elif dataset_type == "val":
            ids = self._val_ids
        elif dataset_type == "test":
            ids = self._test_ids
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

        subset = {id: self.sequences[id] for id in ids}
        dataset = TimeSeriesDataset(subset)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(dataset_type == "train"),  # Shuffle only for training data
        )

    def _split_data(self) -> tuple[list[int], ...]:
        sample_ids = list(self.sequences.keys())
        train_ids, test_ids = train_test_split(
            sample_ids, test_size=self.test_split, random_state=self.random_seed
        )
        train_ids, val_ids = train_test_split(
            train_ids, test_size=self.val_split, random_state=self.random_seed
        )
        return train_ids, val_ids, test_ids

    def _load_csv(self, data_path: str) -> pd.DataFrame:
        return pd.read_csv(data_path)
