import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    """Dataset for time-series or otherwise ordered data with padding at sequence beginnings"""

    def __init__(self, sequences: dict, max_seq_len: int):
        self.sequences = sequences
        self.sample_ids = list(sequences.keys())
        self.max_seq_len = max_seq_len

        self.padded_sequences = {}
        for sample_id, seq_data in sequences.items():
            self.padded_sequences[sample_id] = self._pad_sequence(seq_data)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        padded_sequence = self.padded_sequences[sample_id]
        return padded_sequence, sample_id

    def _pad_sequence(self, sequence_data):
        """Pad a sequence at the beginning to the maximum length"""
        sequence = torch.tensor(sequence_data, dtype=torch.float32)
        seq_len = sequence.shape[0]
        feature_dim = sequence.shape[1]

        if not self.max_seq_len or seq_len >= self.max_seq_len:
            return sequence

        padded_seq = torch.zeros(self.max_seq_len, feature_dim, dtype=torch.float32)
        padding_len = self.max_seq_len - seq_len

        padded_seq[padding_len:] = sequence

        return padded_seq


class SequenceLoader:
    """Loader for sequence data with variable lengths, padded at initialization"""

    def __init__(
        self,
        data_path: str,
        sequence_col: str = "timepoint",
        id_col: str = "sample_id",
        val_split: float = 0.2,
        test_split: float = 0.2,
        batch_size: int = 16,
        max_seq_len: int | None = None,
        random_seed: int = 42,
    ):
        self.data = self._load_csv(data_path)
        self.sequence_col = sequence_col
        self.id_col = id_col
        self.feature_cols = [
            col for col in self.data.columns if col not in [self.sequence_col, self.id_col, "split"]
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

        self.max_seq_len = max_seq_len

        self._train_ids, self._val_ids, self._test_ids = self._split_data()

    def _compute_max_seq_len(self):
        """Compute the maximum sequence length in the dataset"""
        return max(len(seq) for seq in self.sequences.values())

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
        dataset = TimeSeriesDataset(subset, self.max_seq_len)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(dataset_type == "train"),  # Shuffle only for training data
        )

    def _split_data(self) -> tuple[list[str], list[str], list[str]]:
        """Random train/val/test split over all sample ids (ignores any ``split`` column in the CSV)."""
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
