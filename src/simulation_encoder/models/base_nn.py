from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Optional

import neuralop.models as neuralops_models
import torch
from torch import nn
from torch.utils.data import DataLoader

from simulation_encoder.logger import Logger
from simulation_encoder.models.vit import (
    build_vision_transformer,
    build_vision_transformer_decoder,
)

# Layer types that are not in torch.nn
NEURALOP_LAYER_TYPES = ("FNO", "TFNO")
CUSTOM_LAYER_TYPES = (
    "Unflatten",
    "VisionTransformer",
    "VisionTransformerDecoder",
    "PointwiseLinear",
)

_DECODER_NUM_UPSAMPLE_STAGES = 4
_DECODER_INIT_CHANNELS = 64


class PointwiseLinear(nn.Module):
    """1×1 convolution (channel mixing without spatial interaction)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) → (B, H, W, C) → linear → (B, H, W, C') → (B, C', H, W)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.linear(x)
        return x.permute(0, 3, 1, 2).contiguous()


class BaseNN(ABC, nn.Module):
    """
    Abstract base class for autoencoder networks with config-driven definitions.
    """

    @abstractmethod
    def __init__(
        self,
        name: str = "",
        architecture: dict[str, list[dict[str, Any]]] = {},
        num_channels: int = 1,
        num_timepoints: int = 1,
        image_size: int = 128,
        num_epochs: int = 10,
        params: dict[str, Any] = {},
        logger: Optional[Logger] = None,
    ) -> None:
        super().__init__()

    def __str__(self) -> str:
        """Generate a string representation of the model with key parameters."""
        optimizer_type = self.optimizers["combined"].__class__.__name__
        optimizer_params = self.params.get("optimizer", {})
        optimizer_details = ", ".join([f"{key}={value}" for key, value in optimizer_params.items()])

        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder_image_params = sum(
            p.numel() for p in self.decoder_image.parameters() if p.requires_grad
        )
        decoder_timepoint_params = sum(
            p.numel() for p in self.decoder_timepoint.parameters() if p.requires_grad
        )
        total_params = encoder_params + decoder_image_params + decoder_timepoint_params

        return (
            f"Model: {self.name}\n"
            f"Latent Dimension: {self.latent_dim}\n"
            f"Optimizer: {optimizer_type} ({optimizer_details})\n"
            f"Image Size: {self.image_size}\n"
            f"Loss Weights: {self.loss_weights}\n"
            f"Encoder Parameters: {encoder_params:,}\n"
            f"Decoder (Image) Parameters: {decoder_image_params:,}\n"
            f"Decoder (Timepoint) Parameters: {decoder_timepoint_params:,}\n"
            f"Total Parameters: {total_params:,}\n"
        )

    @abstractmethod
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        pretrain: bool = False,
        patience: int = 10,
        min_delta: float = 0.0,
    ) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]]]:
        """Fits the network over the training data for a number of epochs."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Performs encoding and decoding."""
        pass

    @abstractmethod
    def train_one_epoch(
        self,
        train_loader: DataLoader,
    ) -> dict[str, float]:
        """Trains the network for one epoch."""
        pass

    @abstractmethod
    def eval_one_epoch(self, val_loader: DataLoader) -> dict[str, float]:
        """Validates the network during training."""
        pass

    def _base_placeholder_values(self) -> dict[str, Any]:
        flat_size = self.image_size * self.image_size * self.num_channels
        decoder_base_hw = self.image_size // (2**_DECODER_NUM_UPSAMPLE_STAGES)
        decoder_init_ch = _DECODER_INIT_CHANNELS
        return {
            "num_channels": self.num_channels,
            "latent_dim": self.latent_dim,
            "num_timepoints": self.num_timepoints,
            "image_size": self.image_size,
            "num_channels_flat": flat_size,
            "decoder_base_hw": decoder_base_hw,
            "decoder_init_channels": decoder_init_ch,
            "decoder_spatial_flat": decoder_init_ch * decoder_base_hw * decoder_base_hw,
        }

    def _resolve_vision_transformer_head_dim(self, raw_config: dict[str, Any]) -> int:
        """
        Width of the ViT linear head (config key ``out_dim``, or legacy ``latent_dim``).
        Used to expose ``vit_encoder_proj_dim`` for BatchNorm/Linear layers that follow ViT.
        """
        r = deepcopy(raw_config)
        values = self._base_placeholder_values()
        if r.get("type") != "VisionTransformer":
            raise ValueError("Expected VisionTransformer layer config")

        def replace_key(key: str) -> None:
            val = r.get(key)
            if val in values:
                r[key] = values[val]

        replace_key("in_channels")
        replace_key("image_size")
        if "out_dim" not in r:
            if "latent_dim" in r:
                r["out_dim"] = r.pop("latent_dim")
            else:
                r["out_dim"] = self.latent_dim
        else:
            r.pop("latent_dim", None)
        replace_key("out_dim")
        out = r["out_dim"]
        if isinstance(out, str) and out in values:
            out = values[out]
        if not isinstance(out, int):
            raise TypeError(
                "VisionTransformer out_dim must resolve to int, "
                f"got {out!r} in config {raw_config!r}"
            )
        return out

    def _encoder_extra_placeholders(self, encoder_configs: list[dict[str, Any]]) -> dict[str, int]:
        """
        Placeholders derived from the encoder YAML, e.g. ViT head width for post-ViT layers.

        ``vit_encoder_proj_dim`` matches the first VisionTransformer block's head output size.
        """
        for cfg in encoder_configs:
            if cfg.get("type") == "VisionTransformer":
                dim = self._resolve_vision_transformer_head_dim(cfg)
                return {"vit_encoder_proj_dim": dim}
        return {}

    def _resolve_placeholders(
        self,
        config: dict[str, Any],
        extra_placeholders: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Resolve placeholder strings in a layer config. Does not mutate the original."""
        values = {**self._base_placeholder_values(), **(extra_placeholders or {})}
        resolved = deepcopy(config)
        layer_type = resolved.get("type", "")

        def replace(key: str) -> None:
            val = resolved.get(key)
            if val in values:
                resolved[key] = values[val]

        if layer_type in ("Conv2d", "ConvTranspose2d", "PointwiseLinear"):
            for k in ("in_channels", "out_channels"):
                replace(k)
        elif layer_type == "Linear":
            if resolved.get("in_features") == "num_channels":
                resolved["in_features"] = values["num_channels_flat"]
            if resolved.get("out_features") == "num_channels":
                resolved["out_features"] = values["num_channels_flat"]
            replace("in_features")
            replace("out_features")
        elif layer_type in ("BatchNorm1d", "BatchNorm2d"):
            replace("num_features")
            if resolved.get("num_features") == "num_channels":
                resolved["num_features"] = values["num_channels_flat"]
        elif layer_type in NEURALOP_LAYER_TYPES:
            replace("in_channels")
            replace("out_channels")
            resolved.setdefault("n_modes", [16, 16])
            resolved.setdefault("hidden_channel", 64)
        elif layer_type == "VisionTransformer":
            replace("in_channels")
            replace("image_size")

            if "out_dim" not in resolved:
                if "latent_dim" in resolved:
                    resolved["out_dim"] = resolved.pop("latent_dim")
                else:
                    resolved["out_dim"] = self.latent_dim
            else:
                resolved.pop("latent_dim", None)
            replace("out_dim")
            resolved.setdefault("in_channels", self.num_channels)
            resolved.setdefault("image_size", self.image_size)
            resolved.setdefault("out_dim", self.latent_dim)
        elif layer_type == "VisionTransformerDecoder":
            replace("latent_dim")
            replace("image_size")
            replace("in_channels")
            if resolved.get("in_channels") is None:
                resolved["in_channels"] = self.num_channels
            resolved.setdefault("latent_dim", self.latent_dim)
            resolved.setdefault("image_size", self.image_size)
            resolved.setdefault("patch_size", 16)
            resolved.setdefault("embed_dim", 192)
            resolved.setdefault("depth", 6)
            resolved.setdefault("num_heads", 3)
            resolved.setdefault("mlp_ratio", 4.0)
            resolved.setdefault("dropout", 0.0)
        elif layer_type == "Unflatten":
            shape = resolved.get("shape")
            if shape is not None:
                resolved["shape"] = tuple(
                    values.get(s, s) if isinstance(s, str) else s for s in shape
                )
            else:
                resolved["shape"] = (
                    self.num_channels,
                    self.image_size,
                    self.image_size,
                )

        return resolved

    def _build_layer(
        self,
        config: dict[str, Any],
        extra_placeholders: Optional[dict[str, Any]] = None,
    ) -> nn.Module:
        """Build a single layer from a resolved config. Dispatches by layer type."""
        resolved = self._resolve_placeholders(config, extra_placeholders)
        layer_type: str = resolved.get("type", "")

        if layer_type in NEURALOP_LAYER_TYPES:
            layer_class = getattr(neuralops_models, layer_type, None)
            if layer_class is None:
                raise ValueError(f"NeuralOp layer type {layer_type} not found")
            kwargs = {k: v for k, v in resolved.items() if k != "type"}
            return layer_class(**kwargs)

        if layer_type == "VisionTransformer":
            context = {
                "in_channels": resolved.get("in_channels", self.num_channels),
                "image_size": resolved.get("image_size", self.image_size),
                "out_dim": resolved.get("out_dim", self.latent_dim),
            }
            return build_vision_transformer(resolved, context)

        if layer_type == "VisionTransformerDecoder":
            context = {
                "latent_dim": resolved.get("latent_dim", self.latent_dim),
                "image_size": resolved.get("image_size", self.image_size),
                "in_channels": resolved.get("in_channels", self.num_channels),
            }
            return build_vision_transformer_decoder(resolved, context)

        if layer_type == "PointwiseLinear":
            return PointwiseLinear(
                in_channels=resolved["in_channels"],
                out_channels=resolved["out_channels"],
            )

        if layer_type == "Unflatten":
            shape = resolved.get(
                "shape",
                (self.num_channels, self.image_size, self.image_size),
            )
            return nn.Unflatten(1, shape)

        layer_class = getattr(nn, layer_type, None)
        if layer_class is None:
            raise ValueError(
                f"Layer type {layer_type} not recognized. "
                f"Supported: nn.*, {NEURALOP_LAYER_TYPES}, {CUSTOM_LAYER_TYPES}"
            )
        kwargs = {k: v for k, v in resolved.items() if k != "type"}
        return layer_class(**kwargs)

    def _create_layers(
        self,
        layer_configs: list[dict[str, Any]],
        extra_placeholders: Optional[dict[str, Any]] = None,
    ) -> list[nn.Module]:
        """Build a list of modules from a list of layer configs (for encoder/decoder)."""
        layers: list[nn.Module] = []
        for config in layer_configs:
            layer = self._build_layer(config, extra_placeholders)
            layers.append(layer)
        return layers

    def _get_grad_norm(self, module: nn.Module) -> torch.Tensor:
        """Gradient norm of the last parameter that has a gradient (for logging)."""
        device = next(module.parameters()).device
        for m in reversed(list(module.modules())):
            if hasattr(m, "weight") and m.weight.grad is not None:
                return torch.norm(m.weight.grad).to(device)
        return torch.tensor(0.0, device=device)

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _log(self, msg: str, level: str = "info") -> None:
        if self.logger:
            if level == "warning":
                self.logger.warning(msg)
            else:
                self.logger.log(msg)
