from collections.abc import Callable
from typing import Any

import torch
from torchvision import transforms


class Augmentation:
    """
    Class for image augmentation technique.

    Parameters
    ----------
    transform : Callable[[torch.Tensor], torch.Tensor]
        Transformation function to apply to the image tensor.
    name : str
        Name of the augmentation technique.
    """

    def __init__(
        self, transform: Callable[[torch.Tensor], torch.Tensor], name: str
    ) -> None:
        self.transform = transform
        self.name = name

    def __str__(self) -> str:
        return f"{self.name}: {self.transform.__class__.__name__}"

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 3:  # Single image tensor
            return self.transform(tensor)

        if tensor.ndim == 4:  # Stack of image tensors
            transformed_tensors = []
            for i in range(tensor.size(0)):
                transformed_tensors.append(self.transform(tensor[i]))
            return torch.stack(transformed_tensors, dim=0)

        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")


class AugmentationsManager:
    """Handles data augmentaiton"""

    # Additional augmentations can be included here
    AUGMENTATIONS = {
        "identity": lambda _: transforms.Lambda(lambda x: x),
        "rotate": lambda degree: transforms.RandomRotation(degrees=degree),
    }

    def __init__(self, augmentations: list[dict[str, Any]]):
        self.augmentations = augmentations
        self.transforms = self._prepare_augmentations()

    def _prepare_augmentations(self) -> list[dict[str, Augmentation]]:
        transforms_list: list[dict[str, Augmentation]] = []
        if not self.augmentations:
            return transforms_list

        for aug in self.augmentations:
            ((aug_name, arg),) = aug.items()
            if aug_name not in self.AUGMENTATIONS:
                raise ValueError(f"Invalid augmentation: {aug_name}")

            transform = self.AUGMENTATIONS[aug_name](
                int(arg) if arg is not None else arg
            )  # type: ignore
            transforms_list.append({aug_name: Augmentation(transform, aug_name)})

        return transforms_list

    def apply(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        """Applies the specified augmentation."""
        augmented_tensors = [tensor]
        for transform_dict in self.transforms:
            for aug_name, aug in transform_dict.items():
                if aug_name != "identity":
                    augmented_tensors.append(aug(tensor))
        return augmented_tensors
