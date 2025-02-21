from typing import Optional, Callable, Any
from collections import defaultdict

import numpy as np


class DatasetSplitter:
    """Handles dataset splitting while keeping time series grouped together."""

    def __init__(
        self, data: list[dict[str, any]], val_split: float, test_split: float, random_seed: int = 42
    ):
        self.data = data
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed
        self._train_indices = []
        self._val_indices = []
        self._test_indices = []
        self._split_data()

    def _split_data(self):
        """Ensures that all images from a single time series stay in the same split."""
        all_indices = list(range(len(self.data)))
        grouped_indices = defaultdict(list)

        for idx in all_indices:
            key = self.data[idx]["sample_id"]
            grouped_indices[key].append(idx)

        simulation_ids = list(grouped_indices.keys())
        np.random.seed(self.random_seed)
        np.random.shuffle(simulation_ids)

        test_count = int(np.floor(len(simulation_ids) * self.test_split))
        val_count = int(np.floor(len(simulation_ids) * self.val_split))

        test_keys = simulation_ids[:test_count]
        val_keys = simulation_ids[test_count : test_count + val_count]
        train_keys = simulation_ids[test_count + val_count :]

        self._train_indices = [i for key in train_keys for i in grouped_indices[key]]
        self._val_indices = [i for key in val_keys for i in grouped_indices[key]]
        self._test_indices = [i for key in test_keys for i in grouped_indices[key]]

    def get_splits(self):
        return self._train_indices, self._val_indices, self._test_indices
