from typing import Any

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from simulation_encoder.loaders.dataset_utils.augmentation import AugmentationsManager


class ImageDataset(Dataset):
    """Loads images and ensures they are assigned to the correct dataset split."""

    def __init__(
        self,
        image_dir: str,
        data: list[dict[str, Any]],
        channels: list[str],
        augmentation_manager: AugmentationsManager,
        indices: list[int],
        sequence: bool = False,
    ):
        self.image_dir = image_dir
        self.data = [data[i] for i in indices]
        self.channels = channels
        self.augmentation_manager = augmentation_manager
        self.sequence = sequence
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x[:3]),
                transforms.Grayscale(num_output_channels=1),
                transforms.Lambda(lambda x: x.squeeze()),
                transforms.Lambda(lambda x: x / x.max() if x.max() > 0 else x),
            ]
        )

    @property
    def image_shape(self) -> tuple[int, ...]:
        """Shape of the images"""
        if self.sequence:
            # Find first valid image path
            for group in self.data:
                for channel in self.channels:
                    if group[channel]:
                        return Image.open(group[channel][0]).size
            return (128, 128) # Default size
        else:
            return Image.open(self.data[0][self.channels[0]]).size

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_group = self.data[idx]
        
        if self.sequence:
            # For a sequence, we don't have a single timepoint. Returning 0 as a placeholder.
            timepoint = 0
            
            all_channel_tensors = []
            for channel in self.channels:
                image_paths = image_group[channel]
                channel_tensors = []
                for image_path in image_paths:
                    if image_path:
                        image = Image.open(image_path)
                        channel_tensors.append(self.transforms(image))
                    else:
                        channel_tensors.append(torch.zeros(self.image_shape))
                all_channel_tensors.append(torch.stack(channel_tensors, dim=0))
            
            image_tensor = torch.cat(all_channel_tensors, dim=0)

        else:
            timepoint = int(image_group["timepoint"])

            # preprocess
            tensors = [
                (
                    self.transforms(Image.open(image_group[channel]))
                    if image_group[channel]
                    else torch.zeros(self.image_shape)
                )
                for channel in self.channels
            ]
            image_tensor = torch.stack(tensors, dim=0)

        # apply data augmentation to training images
        ((aug_name, augmentation),) = image_group["augmentation"].items()
        if aug_name != "identity":
            image_tensor = augmentation(image_tensor)

        return image_tensor, timepoint