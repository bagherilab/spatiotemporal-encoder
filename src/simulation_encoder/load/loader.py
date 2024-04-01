import os
import numpy as np
import pandas as pd

from PIL import Image

import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms

from sklearn.model_selection import train_test_split

class UnlabeledImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.groups = self.get_image_groups()

    def get_image_groups(self):
        groups = {}
        for filename in os.listdir(self.image_dir):
            if filename.endswith('.png'):
                parts = filename.split('_')
                context = parts[0]  # 'CH' for healthy tissue or 'C' colony
                vasc_type = parts[1] 
                seed = int(parts[2])  # Seed value
                timepoint = int(parts[3])
                if parts[4].split('.')[0] == 'graph':
                    image_type = parts[4].split('.')[0]
                elif parts[4].split('.')[0] == 'cells':
                    image_type = parts[5].split('.')[0]
                else:
                    raise ValueError(f"Invalid name format for image file. Should be \'context_vasc-type_seed_timepoint_image-type.png\' Got: {filename}")
                group_key = (context, vasc_type, seed, timepoint)
                if group_key not in groups:
                    groups[group_key] = {'cancer': None, 'healthy': None, 'graph': None}
                groups[group_key][image_type] = os.path.join(self.image_dir, filename)
        return list(groups.values())

    def resize(self, group):
        cancer_img = Image.open(group['cancer'])
        healthy_img = Image.open(group['healthy'])
        graph_img = Image.open(group['graph'])
        width, height = max(cancer_img.size, healthy_img.size, graph_img.size)
        cancer_img = cancer_img.resize((width, height))
        healthy_img = healthy_img.resize((width, height))
        graph_img = graph_img.resize((width, height))
        return cancer_img, healthy_img, graph_img
    
    def display_image(self, image):
        image = np.squeeze(image)
        image = (image * 255).astype('uint8')
        Image.fromarray(image).show()

    def display_tensor(self, tensor):
        array = np.moveaxis(tensor.numpy(), 0, -1)
        array = array.squeeze()
        image = Image.fromarray(array.astype('uint8'))
        image.save('output_image.png')

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]
        cancer_path = group['cancer']
        healthy_path = group['healthy']
        graph_path = group['graph']

        transformation = transforms.Compose([
            transforms.ToTensor()
        ])

        cancer_tensor = transformation(Image.open(cancer_path).convert('L')).squeeze()
        healthy_tensor = transformation(Image.open(healthy_path).convert('L')).squeeze()
        graph_tensor =  transformation(Image.open(graph_path).convert('L')).squeeze()

        return torch.stack((cancer_tensor, healthy_tensor, graph_tensor), dim=0)

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
            self.graph_dfs = [graph_df.drop(
            columns=["avg_eccentricity_weighted", "avg_closeness_weighted", "avg_coreness_weighted", "avg_betweenness_weighted", "avg_out_degrees_weighted", "avg_in_degrees_weighted", "avg_degree_weighted"]
            ) for graph_df in self.graph_dfs]

        self.feature_columns = None
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
        
        self.feature_columns = self.features.columns
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
