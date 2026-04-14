import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def last_adaptive_avg_pool2d_in_module(module: nn.Module) -> nn.AdaptiveAvgPool2d | None:
    """Return the last ``AdaptiveAvgPool2d`` in depth-first order (typically the final global pool)."""
    last: nn.AdaptiveAvgPool2d | None = None
    for m in module.modules():
        if isinstance(m, nn.AdaptiveAvgPool2d):
            last = m
    return last


def _find_vision_transformer(module: nn.Module):
    """Return the first ``VisionTransformer`` sub-module (imported lazily to avoid circular deps)."""
    from simulation_encoder.models.vit import VisionTransformer

    for m in module.modules():
        if isinstance(m, VisionTransformer):
            return m
    return None


def gradcam_map_pre_adaptive_avg_pool2d(
    activations: torch.Tensor,
    grad_wrt_activations: torch.Tensor,
    out_hw: tuple[int, int],
) -> torch.Tensor:
    """
    Standard Grad-CAM from feature map ``activations`` and loss gradient w.r.t. those features.

    Parameters
    ----------
    activations
        (B, C, H, W) — input to ``AdaptiveAvgPool2d``.
    grad_wrt_activations
        Same shape as ``activations``.
    out_hw
        Spatial size (H, W) to upsample to (usually input image size).
    """
    weights = grad_wrt_activations.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1)
    cam = F.relu(cam)
    cam = cam.unsqueeze(1)
    cam = F.interpolate(cam, size=out_hw, mode="bilinear", align_corners=False).squeeze(1)
    return cam


def gradcam_map_from_vit_tokens(
    activations: torch.Tensor,
    grad_wrt_activations: torch.Tensor,
    out_hw: tuple[int, int],
) -> torch.Tensor:
    """
    Grad-CAM for ViT: the ``norm`` layer output is ``(B, N, E)`` where ``N = H_p * W_p`` patches.

    Reshape tokens to a spatial grid ``(B, E, H_p, W_p)``, then apply the same channel-weighted
    sum as the CNN variant.
    """
    B, N, E = activations.shape
    H_p = int(math.isqrt(N))
    W_p = N // H_p
    A = activations.permute(0, 2, 1).reshape(B, E, H_p, W_p)
    G = grad_wrt_activations.permute(0, 2, 1).reshape(B, E, H_p, W_p)
    return gradcam_map_pre_adaptive_avg_pool2d(A, G, out_hw)


def reconstruction_saliency_map(
    model: nn.Module,
    encoder: nn.Module,
    x: torch.Tensor,
    image_criterion: nn.Module,
    get_pred_image: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """
    Grad-CAM for reconstruction loss (MSE on decoded image vs input).

    Uses the last ``AdaptiveAvgPool2d`` input in ``encoder`` if present (CNN / FNO),
    else the last ``VisionTransformer`` ``norm`` token output, else max |∂L/∂x| over channels.
    """
    model.eval()

    x = x.detach().clone().requires_grad_(True)

    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}
    hooks: list = []
    cam_source: str = "input"

    pool = last_adaptive_avg_pool2d_in_module(encoder)
    vit = _find_vision_transformer(encoder) if pool is None else None

    if pool is not None:
        cam_source = "pool"

        def _fwd_hook(_m: nn.Module, inp: tuple[torch.Tensor, ...], _out: torch.Tensor) -> None:
            activations["a"] = inp[0]

        def _bwd_hook(
            _m: nn.Module,
            grad_input: tuple[torch.Tensor | None, ...],
            _grad_output: tuple[torch.Tensor, ...],
        ) -> None:
            if grad_input[0] is not None:
                gradients["g"] = grad_input[0]

        hooks.append(pool.register_forward_hook(_fwd_hook))
        hooks.append(pool.register_full_backward_hook(_bwd_hook))

    elif vit is not None:
        cam_source = "vit"
        target = vit.norm

        def _fwd_hook_vit(_m: nn.Module, inp: tuple[torch.Tensor, ...], _out: torch.Tensor) -> None:
            activations["a"] = inp[0]

        def _bwd_hook_vit(
            _m: nn.Module,
            grad_input: tuple[torch.Tensor | None, ...],
            _grad_output: tuple[torch.Tensor, ...],
        ) -> None:
            if grad_input[0] is not None:
                gradients["g"] = grad_input[0]

        hooks.append(target.register_forward_hook(_fwd_hook_vit))
        hooks.append(target.register_full_backward_hook(_bwd_hook_vit))

    out_hw = (x.shape[2], x.shape[3])
    saliency_map = torch.zeros_like(x[:, 0, :, :])

    try:
        pred_image = get_pred_image(x)
        loss_image = image_criterion(pred_image, x)
        loss_image.backward()

        if cam_source == "pool" and "a" in activations and "g" in gradients:
            A, G = activations["a"], gradients["g"]
            if A.shape == G.shape:
                return gradcam_map_pre_adaptive_avg_pool2d(A, G, out_hw)

        if cam_source == "vit" and "a" in activations and "g" in gradients:
            A, G = activations["a"], gradients["g"]
            if A.shape == G.shape:
                return gradcam_map_from_vit_tokens(A, G, out_hw)

        if x.grad is not None:
            saliency_map, _ = torch.max(x.grad.abs(), dim=1)
        else:
            raise RuntimeError("Gradient is None")
    except RuntimeError as e:
        print(f"Error during saliency map computation: {e}")
        saliency_map = torch.zeros_like(x[:, 0, :, :])
    finally:
        for h in hooks:
            h.remove()

    return saliency_map
