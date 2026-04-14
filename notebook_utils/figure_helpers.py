"""Paper figures: data + matplotlib helpers"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.decomposition import PCA

from notebook_utils.embedding_helpers import (
    load_encoded_latents,
    load_encoded_metadata,
    timepoint_decoder_true_pred,
)

def plot_figure2b_spatial_mse_bars(
    rows: list[dict[str, Any]],
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (8.0, 8.0),
    spine_linewidth: float = 1.0,
    yticks: Sequence[float] | None = None,
    ytick_labels: Sequence[str] | None = None,
    font_size: float = 20.0,
    font_family: str | None = "Helvetica Neue",
    ) -> plt.Figure:
    if not rows:
        raise ValueError("no rows to plot")

    order_parts: list[str] = []
    for r in rows:
        name = str(r.get("model", r.get("label", "?")))
        ds = r.get("dataset")
        order_parts.append(f"{name} ({ds})" if ds is not None else name)
    print("Bar order (left → right):", " | ".join(order_parts))

    train_mse_per_group = np.array([r["train_mse"] for r in rows], dtype=float)
    test_mse_per_group = np.array([r["test_mse"] for r in rows], dtype=float)

    n_groups = len(rows)
    bar_group_x = np.arange(n_groups)
    bar_width = 0.36
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.bar(
        bar_group_x - bar_width / 2,
        train_mse_per_group,
        width=bar_width,
        facecolor="none",
        edgecolor="black",
        linewidth=2.5,
    )
    ax.bar(
        bar_group_x + bar_width / 2,
        test_mse_per_group,
        width=bar_width,
        facecolor="#808080",
        edgecolor="black",
        linewidth=2.5,
    )
    ax.grid(False)
    ax.tick_params(axis="both", which="both", length=0, labelsize=font_size)
    ax.tick_params(axis="x", labelbottom=False)
    if yticks is not None:
        yticks_list = list(yticks)
        if ytick_labels is not None:
            ytick_labels_list = list(ytick_labels)
            if len(ytick_labels_list) != len(yticks_list):
                raise ValueError(
                    "ytick_labels length must match yticks "
                    f"({len(ytick_labels_list)} != {len(yticks_list)})"
                )
            ax.set_yticks(yticks_list, labels=ytick_labels_list)
        else:
            ax.set_yticks(yticks_list)
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        if font_family is not None:
            label.set_fontfamily(font_family)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        sp = ax.spines[side]
        sp.set_visible(True)
        sp.set_linewidth(spine_linewidth)
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig

def plot_figure2c_timepoint_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figsize: tuple[float, float] = (8.0, 8.0),
    base_dot_size: float = 36.0,
    count_size_power: float = 1.0,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Parity plot: predicted vs true timepoint class indices (decoder argmax vs loader labels).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    pairs = np.stack([y_true.astype(np.int64, copy=False), y_pred.astype(np.int64, copy=False)], axis=1)
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)
    ux = uniq[:, 0].astype(float)
    uy = uniq[:, 1].astype(float)
    sizes = base_dot_size * (counts.astype(float) ** count_size_power)

    sns.scatterplot(
        x=ux,
        y=uy,
        s=sizes,
        ax=ax,
        color="#83ABCF",
        alpha=1.0,
        linewidth=1.5,
        edgecolor="#6F96B8",
        legend=False,
    )
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    span = hi - lo
    pad = 0.05 * span
    lim_lo, lim_hi = lo - pad, hi + pad
    ax.plot(
        [lim_lo, lim_hi],
        [lim_lo, lim_hi],
        color="black",
        linewidth=1.0,
        linestyle="--",
    )
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)
    ax.set_frame_on(False)
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_visible(False)
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig, ax


_WHITE_RED_CMAP = LinearSegmentedColormap.from_list(
    "figure5b_white_red", [(1.0, 1.0, 1.0), (1.0, 0.0, 0.0)]
)


def _figure5b_saliency_rgba(
    sal_norm: np.ndarray,
    saliency_alpha: float,
    *,
    gamma: float = 1.0,
    white_alpha_floor: float = 0.22,
) -> np.ndarray:
    """
    White→red saliency as RGBA on top of the grayscale underlay.

    ``white_alpha_floor`` (0–1) sets the minimum opacity **at zero saliency**
    (white end of the map), scaled by ``saliency_alpha``, so white is not fully
    transparent. Opacity ramps to ``saliency_alpha`` at full saliency (red).

    Optional ``gamma`` is applied to normalized saliency before colormap/alpha
    (``1.0`` = linear).
    """
    t = np.clip(np.asarray(sal_norm, dtype=np.float64), 0.0, 1.0)
    g = float(gamma)
    if g > 0.0 and g != 1.0:
        t = np.power(t, g)
    rgba = np.asarray(_WHITE_RED_CMAP(t), dtype=np.float64)
    rgba = np.clip(rgba, 0.0, 1.0)
    floor = float(np.clip(white_alpha_floor, 0.0, 1.0))
    a_scale = floor + (1.0 - floor) * t
    rgba[..., 3] = np.clip(float(saliency_alpha) * a_scale, 0.0, 1.0)
    return rgba


def _figure5b_spatial_shape(arr: np.ndarray) -> tuple[int, int]:
    if arr.ndim == 2:
        return int(arr.shape[0]), int(arr.shape[1])
    return int(arr.shape[0]), int(arr.shape[1])


def _figure5b_figsize_and_dpi(arr: np.ndarray, max_display_inches: float) -> tuple[float, float, float]:
    """
    Figure size (inches) and DPI so ``fig.get_figwidth() * dpi ≈ W`` (same for H).

    Matches canvas pixels to image pixels to avoid upscaling with a white mat in
    Jupyter. ``max_display_inches`` is the longer side of the figure in inches.
    """
    h, w = _figure5b_spatial_shape(arr)
    if h <= 0 or w <= 0 or max_display_inches <= 0:
        return (4.0, 4.0, 100.0)
    long_px = max(h, w)
    dpi = float(long_px) / float(max_display_inches)
    return (w / dpi, h / dpi, dpi)


def _figure5b_imshow_limits(ax: plt.Axes, arr: np.ndarray) -> None:
    """Pin limits to pixel grid (origin upper) so no extra axis padding around the image."""
    h, w = _figure5b_spatial_shape(arr)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)


def _figure5b_new_figure(fig_w: float, fig_h: float, dpi: float) -> tuple[plt.Figure, plt.Axes]:
    """
    Borderless figure matching Fig 2a in ``figure_notebook``: ``frameon=False`` and a
    single axes that fills the canvas (no subplot margins).
    """
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, frameon=False)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.patch.set_alpha(0.0)
    ax.set_axis_off()
    return fig, ax


def _savefig_transparent(fig: plt.Figure, path: Path, *, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=dpi,
        transparent=True,
        facecolor="none",
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0,
    )


_figure5b_inline_display_configured = False


def configure_figure5b_inline_display(*, force: bool = False) -> None:
    """
    Jupyter / IPython: tighten how inline figures are rasterized (removes white mat).

    Call once in a notebook (or rely on ``plot_figure5b_saliency_row``, which calls
    this automatically). Safe no-op outside IPython.
    """
    global _figure5b_inline_display_configured
    if _figure5b_inline_display_configured and not force:
        return
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is None:
            return
        cur = dict(getattr(ip.config.InlineBackend, "print_figure_kwargs", {}) or {})
        cur.update(
            {
                "bbox_inches": "tight",
                "pad_inches": 0.0,
                "facecolor": "none",
                "transparent": True,
            }
        )
        ip.config.InlineBackend.print_figure_kwargs = cur
        _figure5b_inline_display_configured = True
    except Exception:
        pass


def show_figure5b_figures(figures: list[plt.Figure]) -> None:
    """
    Show Figure 5b outputs in Jupyter **without** the inline backend's extra mat.

    Rasterizes each figure the same way as ``savefig`` (tight, transparent) and
    displays PNG bytes. Falls back to ``plt.show()`` outside IPython.
    Closes each figure afterwards to prevent the inline backend from
    auto-displaying it a second time at the end of the cell.
    """
    if not figures:
        return
    try:
        import io

        from IPython.display import Image, display

        for fig in figures:
            buf = io.BytesIO()
            fig.savefig(
                buf,
                format="png",
                dpi=fig.dpi,
                bbox_inches="tight",
                pad_inches=0.0,
                facecolor="none",
                edgecolor="none",
                transparent=True,
            )
            buf.seek(0)
            display(Image(data=buf.getvalue()))
            plt.close(fig)
    except Exception:
        for fig in figures:
            plt.figure(fig.number)
        plt.show()


def plot_figure5b_saliency_row(
    dataset: Any,
    sample_index: int,
    models: dict[str, torch.nn.Module],
    *,
    device: torch.device | str = "cpu",
    figsize: tuple[float, float] = (4.0, 4.0),
    underlay_alpha: float = 1.0,
    saliency_alpha: float = 0.55,
    saliency_gamma: float = 1.0,
    saliency_white_alpha_floor: float = 0.22,
    save_path_prefix: str | Path | None = None,
    save_dpi: int = 300,
    configure_inline_display: bool = True,
) -> list[plt.Figure]:
    """
    Figure 5b: original + per-model saliency overlays.

    Saliency uses a **white→red** colormap on top of the image. ``saliency_white_alpha_floor``
    keeps low-saliency (white) pixels from being fully transparent so the map
    stays visible over the gray underlay. ``saliency_alpha`` scales the whole overlay
    (peak opacity at full saliency is ``saliency_alpha``).
    """
    if not models:
        raise ValueError("models must be non-empty")

    if configure_inline_display:
        configure_figure5b_inline_display()

    dev = torch.device(device)
    image, _ = dataset[sample_index]
    if not torch.is_tensor(image):
        image = torch.as_tensor(image)
    image = image.to(dev).float()

    orig = image.detach().cpu()
    if orig.ndim == 3 and orig.shape[0] == 1:
        orig_display = orig.squeeze(0).numpy()
    elif orig.ndim == 3:
        orig_display = orig.permute(1, 2, 0).numpy()
    else:
        orig_display = orig.numpy()

    figures: list[plt.Figure] = []
    save_base = Path(save_path_prefix) if save_path_prefix is not None else None
    max_in = float(max(figsize))
    f_w, f_h, disp_dpi = _figure5b_figsize_and_dpi(orig_display, max_in)

    fig_o, ax_o = _figure5b_new_figure(f_w, f_h, disp_dpi)
    ax_o.imshow(
        orig_display,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        alpha=float(underlay_alpha),
        aspect="auto",
    )
    _figure5b_imshow_limits(ax_o, orig_display)
    figures.append(fig_o)

    if save_base is not None:
        _savefig_transparent(
            fig_o, save_base.parent / f"{save_base.name}_original.png", dpi=save_dpi
        )

    for label, model in models.items():
        model = model.to(dev)
        model.eval()
        model.zero_grad(set_to_none=True)
        inp = image.unsqueeze(0).detach().clone()
        with torch.enable_grad():
            sal_t = model.get_saliency_map(inp)
        sal = sal_t.squeeze().detach().cpu().numpy()
        smin, smax = float(np.min(sal)), float(np.max(sal))
        if smax > smin:
            sal_norm = (sal - smin) / (smax - smin)
        else:
            sal_norm = np.zeros_like(sal)

        fig_m, ax_m = _figure5b_new_figure(f_w, f_h, disp_dpi)
        ax_m.imshow(
            orig_display,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            alpha=float(underlay_alpha),
            interpolation="nearest",
            aspect="auto",
        )
        sal_rgba = _figure5b_saliency_rgba(
            sal_norm,
            float(saliency_alpha),
            gamma=float(saliency_gamma),
            white_alpha_floor=float(saliency_white_alpha_floor),
        )
        ax_m.imshow(sal_rgba, interpolation="nearest", aspect="auto")
        _figure5b_imshow_limits(ax_m, orig_display)
        figures.append(fig_m)

        if save_base is not None:
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
            _savefig_transparent(
                fig_m, save_base.parent / f"{save_base.name}_{safe}.png", dpi=save_dpi
            )

    return figures


def _min_train_image_mse(results_json: Path) -> float | None:
    with open(results_json, encoding="utf-8") as f:
        data = json.load(f)
    series = data.get("losses", {}).get("train", {}).get("image", [])
    if not series:
        return None
    return float(min(series))

def _tensors_to_cpu(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.cpu()
    if isinstance(x, tuple):
        return tuple(_tensors_to_cpu(v) for v in x)
    if isinstance(x, list):
        return [_tensors_to_cpu(v) for v in x]
    if isinstance(x, dict):
        return {k: _tensors_to_cpu(v) for k, v in x.items()}
    return x

def collect_spatial_mse_train_test(
    results_dir: str | Path,
    all_models: dict[str, dict[str, torch.nn.Module]],
    data_loaders: dict[str, dict[str, Any]],
    device: torch.device | str = "cpu",
    ) -> list[dict[str, Any]]:
    results_path = Path(results_dir)
    rows: list[dict[str, Any]] = []

    for model_name, datasets in sorted(all_models.items()):
        for dataset_name, model in sorted(datasets.items()):
            results_file = results_path / model_name / "_best_model" / dataset_name / "results.json"
            train_mse = _min_train_image_mse(results_file)
            test_loader = _CpuBatchDataLoader(data_loaders[dataset_name]["test"])
            model.eval()
            dev = torch.device(device)
            model.to(dev)
            if hasattr(model, "device"):
                model.device = str(dev)
            with torch.no_grad():
                metrics = model.eval_one_epoch(test_loader)
            test_mse = float(metrics["image"])
            rows.append(
                {
                    "model": model_name,
                    "dataset": dataset_name,
                    "label": model_name,
                    "train_mse": train_mse,
                    "test_mse": test_mse,
                }
            )
    return rows

def collect_timepoint_parity_data(
    results_dir: str | Path,
    all_models: dict[str, dict[str, torch.nn.Module]],
    split: str = "test",
    device: torch.device | str = "cpu",
    ) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """
    True vs predicted timepoint class index per model/dataset.
    """
    embeddings = load_encoded_latents(results_dir, all_models=all_models, split=split)
    true_tp, _ = load_encoded_metadata(results_dir, all_models=all_models, split=split)
    return timepoint_decoder_true_pred(all_models, embeddings, true_tp, device=device)

class _CpuBatchDataLoader:
    """Iterates like a DataLoader but moves all tensors in each batch to CPU."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader

    def __iter__(self) -> Iterator[Any]:
        for batch in self._loader:
            yield _tensors_to_cpu(batch)

    def __len__(self) -> int:
        return len(self._loader)


# ---------------------------------------------------------------------------
# Figure 5c — PC perturbation difference images
# ---------------------------------------------------------------------------

def rank_pcs_by_reconstruction_importance(
    model: torch.nn.Module,
    latents: torch.Tensor,
    *,
    n_components: int | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """
    Fit PCA on ``latents`` and rank components by how much ``decode_image``
    output changes when moving ±1 σ along each PC from the mean latent.

    Returns a dict with keys:

    * ``pca`` — fitted ``sklearn.decomposition.PCA``
    * ``z_mean`` — (D,) mean latent
    * ``explained_variance_ratio`` — per-PC fraction of latent variance
    * ``reconstruction_importance`` — (K,) L2 sensitivity of decoder per PC
    * ``importance_order`` — PC indices sorted by reconstruction_importance descending
    """
    dev = torch.device(device)
    Z = latents.detach().cpu().numpy().astype(np.float64)
    n = n_components or Z.shape[1]

    pca = PCA(n_components=n)
    pca.fit(Z)

    z_mean_np = pca.mean_.astype(np.float32)
    z_mean = torch.from_numpy(z_mean_np).to(dev)

    model.to(dev).eval()
    with torch.no_grad():
        ref_img = model.decode_image(z_mean.unsqueeze(0)).squeeze(0)

    importance = np.zeros(n, dtype=np.float64)
    for k in range(n):
        sigma_k = float(np.sqrt(pca.explained_variance_[k]))
        v_k = torch.from_numpy(pca.components_[k].astype(np.float32)).to(dev)
        with torch.no_grad():
            img_plus = model.decode_image((z_mean + sigma_k * v_k).unsqueeze(0)).squeeze(0)
            img_minus = model.decode_image((z_mean - sigma_k * v_k).unsqueeze(0)).squeeze(0)
        diff_plus = (img_plus - ref_img).float()
        diff_minus = (img_minus - ref_img).float()
        importance[k] = float(diff_plus.pow(2).sum().sqrt() + diff_minus.pow(2).sum().sqrt())

    order = np.argsort(importance)[::-1].copy()

    return {
        "pca": pca,
        "z_mean": z_mean_np,
        "explained_variance_ratio": pca.explained_variance_ratio_.copy(),
        "reconstruction_importance": importance,
        "importance_order": order,
    }


def generate_pc_perturbation_diffs(
    model: torch.nn.Module,
    pca_result: dict[str, Any],
    pc_indices: Sequence[int],
    sigma_scale: float = 3.0,
    device: torch.device | str = "cpu",
    base_latent: np.ndarray | torch.Tensor | None = None,
    ) -> dict[str, Any]:
    """
    For each selected PC, perturb a base latent by ±``sigma_scale`` σ along that PC,
    decode through ``decode_image``, and subtract the reference (decoded base).
    """
    dev = torch.device(device)
    pca: PCA = pca_result["pca"]
    z_mean_np = pca_result["z_mean"]
    if base_latent is None:
        z0 = torch.from_numpy(z_mean_np.astype(np.float32)).to(dev)
    else:
        if isinstance(base_latent, torch.Tensor):
            z0 = base_latent.to(device=dev, dtype=torch.float32).reshape(-1)
        else:
            z0 = torch.from_numpy(np.asarray(base_latent, dtype=np.float32)).to(dev).reshape(-1)
        if z0.shape[0] != z_mean_np.shape[0]:
            raise ValueError(
                f"base_latent length {z0.shape[0]} != latent dim {z_mean_np.shape[0]}"
            )

    model.to(dev).eval()
    with torch.no_grad():
        ref_img = model.decode_image(z0.unsqueeze(0)).squeeze(0).cpu().numpy()

    diffs: list[dict[str, Any]] = []
    for k in pc_indices:
        sigma_k = float(np.sqrt(pca.explained_variance_[k]))
        v_k = torch.from_numpy(pca.components_[k].astype(np.float32)).to(dev)
        with torch.no_grad():
            img_p = model.decode_image((z0 + sigma_scale * sigma_k * v_k).unsqueeze(0))
            img_m = model.decode_image((z0 - sigma_scale * sigma_k * v_k).unsqueeze(0))
        diff_p = img_p.squeeze(0).cpu().numpy() - ref_img
        diff_m = img_m.squeeze(0).cpu().numpy() - ref_img
        diffs.append({
            "pc": k,
            "var_explained": float(pca.explained_variance_ratio_[k]),
            "importance": float(pca_result["reconstruction_importance"][k]),
            "diff_plus": diff_p,
            "diff_minus": diff_m,
        })

    return {"reference_image": ref_img, "diffs": diffs}


def _fig5c_to_2d(arr: np.ndarray) -> np.ndarray:
    """(C,H,W) or (H,W,C) diff array → 2-D for imshow."""
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.squeeze(0) if arr.shape[0] == 1 else np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    return arr


def plot_figure5c_pc_diffs(
    diff_result: dict[str, Any],
    *,
    vmax: float | None = None,
    gap_px: int = 4,
    max_display_inches: float = 4.0,
    save_path_prefix: str | Path | None = None,
    save_dpi: int = 300,
    ) -> list[plt.Figure]:
    diffs = diff_result["diffs"]

    if vmax is None:
        all_abs = []
        for d in diffs:
            all_abs.append(np.abs(d["diff_plus"]).max())
            all_abs.append(np.abs(d["diff_minus"]).max())
        vmax = float(max(all_abs)) if all_abs else 1.0

    cmap = plt.cm.RdBu_r  # type: ignore[attr-defined]
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    save_base = Path(save_path_prefix) if save_path_prefix is not None else None
    figures: list[plt.Figure] = []

    for d in diffs:
        img_m = _fig5c_to_2d(d["diff_minus"])
        img_p = _fig5c_to_2d(d["diff_plus"])
        H, W = img_m.shape

        gap = np.full((H, gap_px), np.nan)
        composite = np.concatenate([img_m, gap, img_p], axis=1)
        total_w = 2 * W + gap_px

        long_px = max(H, total_w)
        dpi = float(long_px) / max_display_inches
        fig_w = total_w / dpi
        fig_h = H / dpi

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, frameon=False)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        fig.patch.set_facecolor("none")
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
        ax.patch.set_alpha(0.0)
        ax.set_axis_off()

        cmap_copy = cmap.copy()
        cmap_copy.set_bad(color="white")
        ax.imshow(composite, cmap=cmap_copy, norm=norm, interpolation="nearest", aspect="auto")
        ax.set_xlim(-0.5, total_w - 0.5)
        ax.set_ylim(H - 0.5, -0.5)

        figures.append(fig)

        if save_base is not None:
            pc_k = d["pc"]
            out = save_base.parent / f"{save_base.name}_pc{pc_k + 1}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                out, dpi=save_dpi, transparent=True, facecolor="none",
                edgecolor="none", bbox_inches="tight", pad_inches=0,
            )

    return figures

def _radial_average_psd(image_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Radially-averaged power spectral density.
    """
    H, W = image_2d.shape
    F = np.fft.fft2(image_2d)
    F = np.fft.fftshift(F)
    power = np.abs(F) ** 2

    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)

    max_r = min(cy, cx)
    psd = np.zeros(max_r, dtype=np.float64)
    counts = np.zeros(max_r, dtype=np.float64)
    for ri in range(max_r):
        mask = r == ri
        psd[ri] = power[mask].sum()
        counts[ri] = mask.sum()
    nonzero = counts > 0
    psd[nonzero] /= counts[nonzero]

    freq = np.arange(max_r) / float(max(H, W))
    return freq, psd


def compute_pc_perturbation_psd(
    diff_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Radially-averaged PSD of every PC perturbation diff image.

    Returns a dict with:

    * ``freq`` — 1-D array, spatial frequency in cycles / pixel
    * ``per_pc`` — list of dicts, one per PC, each containing ``pc``,
      ``psd_plus``, ``psd_minus``, ``psd_mean`` (average of both directions)
    * ``mean_psd`` — PSD averaged across all PCs and both directions
    """
    diffs = diff_result["diffs"]
    per_pc: list[dict[str, Any]] = []
    freq: np.ndarray | None = None
    accum: np.ndarray | None = None

    for d in diffs:
        img_p = _fig5c_to_2d(d["diff_plus"])
        img_m = _fig5c_to_2d(d["diff_minus"])

        fp, psd_p = _radial_average_psd(img_p)
        fm, psd_m = _radial_average_psd(img_m)

        if freq is None:
            freq = fp
            accum = np.zeros_like(fp)
        psd_avg = (psd_p + psd_m) / 2.0
        accum += psd_avg  # type: ignore[operator]
        per_pc.append({
            "pc": d["pc"],
            "psd_plus": psd_p,
            "psd_minus": psd_m,
            "psd_mean": psd_avg,
        })

    if accum is not None and len(per_pc):
        mean_psd = accum / len(per_pc)
    else:
        mean_psd = np.array([])

    return {"freq": freq if freq is not None else np.array([]), "per_pc": per_pc, "mean_psd": mean_psd}


def _psd_band_metrics(
    freq: np.ndarray,
    psd: np.ndarray,
    low_freq_max: float = 0.05,
    high_freq_min: float = 0.15,
) -> dict[str, float]:
    """
    Summary statistics from a single radial PSD curve (freq in cycles/pixel, psd ≥ 0).

    Uses trapezoidal integration in ``freq``. Excludes the DC bin (``freq == 0``)
    from centroid / median so the scale is dominated by structure, not the mean offset.
    """
    freq = np.asarray(freq, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)
    valid = (freq > 0) & np.isfinite(psd) & (psd >= 0)
    f = freq[valid]
    p = psd[valid]
    if f.size == 0 or float(np.sum(p)) <= 0:
        return {
            "centroid_freq_cy_per_px": float("nan"),
            "centroid_wavelength_px": float("nan"),
            "median_wavelength_px": float("nan"),
            "frac_power_coarse": float("nan"),
            "frac_power_mid": float("nan"),
            "frac_power_fine": float("nan"),
            "spectral_entropy_bits": float("nan"),
        }

    total = float(np.trapz(p, f))
    if total <= 0:
        total = float(np.sum(p))

    # Centroid frequency (weighted mean f)
    centroid_f = float(np.trapz(f * p, f) / total)
    centroid_wl = 1.0 / centroid_f if centroid_f > 0 else float("nan")

    # Median frequency: half of integrated power below this f (then wavelength = 1/f)
    df = np.diff(f)
    segments = 0.5 * (p[:-1] + p[1:]) * df
    cum = np.concatenate([[0.0], np.cumsum(segments)])
    if cum[-1] > 0:
        cum /= cum[-1]
    idx = int(np.searchsorted(cum, 0.5, side="right"))
    idx = min(max(idx, 1), len(cum) - 1)
    f_lo, f_hi = f[idx - 1], f[idx]
    c_lo, c_hi = cum[idx - 1], cum[idx]
    if c_hi > c_lo:
        t = (0.5 - c_lo) / (c_hi - c_lo)
        f_median = float(f_lo + t * (f_hi - f_lo))
    else:
        f_median = float(f[idx])
    median_wl = 1.0 / f_median if f_median > 0 else float("nan")

    # Band fractions of integrated power (coarse / mid / fine spatial scales)
    mask_l = f <= low_freq_max
    mask_h = f >= high_freq_min
    mask_m = (~mask_l) & (~mask_h)

    def _frac(m: np.ndarray) -> float:
        if not np.any(m):
            return 0.0
        return float(np.trapz(p[m], f[m]) / total)

    frac_coarse = _frac(mask_l)
    frac_fine = _frac(mask_h)
    frac_mid = max(0.0, 1.0 - frac_coarse - frac_fine)

    # Normalized entropy across frequency bins (Shannon, bits)
    p_n = p / np.sum(p)
    p_n = p_n[p_n > 0]
    entropy_bits = float(-np.sum(p_n * np.log2(p_n)))

    return {
        "centroid_freq_cy_per_px": centroid_f,
        "centroid_wavelength_px": centroid_wl,
        "median_wavelength_px": median_wl,
        "frac_power_coarse": frac_coarse,
        "frac_power_mid": frac_mid,
        "frac_power_fine": frac_fine,
        "spectral_entropy_bits": entropy_bits,
    }


def _sample_std(a: Sequence[float] | np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    return float(np.std(x, ddof=1))


def summarize_pc_perturbation_psd_table(
    psd_results: dict[str, dict[str, Any]],
    *,
    low_freq_max: float = 0.05,
    high_freq_min: float = 0.15,
) -> pd.DataFrame:
    """
    Table of spatial-frequency summaries for PC perturbation **difference images**
    (not a single dataset example image).

    For each principal component we decode ``±`` moves in latent space, subtract the
    reference image, build a radial PSD of each diff map, then compute power-weighted
    **mean frequency** (centroid, cycles/pixel) and **coarse / mid / fine** band
    fractions.  Values in the table are the **mean and sample standard deviation of
    those metrics across the top-``N`` PCs** (``N`` = number of PCs in panel c).

    Coarse / mid / fine use integrated PSD with ``f ≤ low_freq_max``,
    between ``low_freq_max`` and ``high_freq_min``, and ``f ≥ high_freq_min``.
    """
    rows: list[dict[str, Any]] = []
    for label, res in psd_results.items():
        freq = res["freq"]
        per_pc = res.get("per_pc") or []

        centroids: list[float] = []
        coarse_l: list[float] = []
        mid_l: list[float] = []
        fine_l: list[float] = []

        for entry in per_pc:
            m = _psd_band_metrics(
                freq,
                entry["psd_mean"],
                low_freq_max=low_freq_max,
                high_freq_min=high_freq_min,
            )
            centroids.append(m["centroid_freq_cy_per_px"])
            coarse_l.append(m["frac_power_coarse"])
            mid_l.append(m["frac_power_mid"])
            fine_l.append(m["frac_power_fine"])

        if not per_pc:
            m = _psd_band_metrics(
                freq, res["mean_psd"],
                low_freq_max=low_freq_max, high_freq_min=high_freq_min,
            )
            centroids = [m["centroid_freq_cy_per_px"]]
            coarse_l = [m["frac_power_coarse"]]
            mid_l = [m["frac_power_mid"]]
            fine_l = [m["frac_power_fine"]]

        rows.append({
            "model": label,
            "n_pcs": len(per_pc) if per_pc else 1,
            "centroid_freq_mean_cy_per_px": float(np.nanmean(centroids)),
            "centroid_freq_std_cy_per_px": _sample_std(centroids),
            "frac_coarse_mean": float(np.nanmean(coarse_l)),
            "frac_coarse_std": _sample_std(coarse_l),
            "frac_mid_mean": float(np.nanmean(mid_l)),
            "frac_mid_std": _sample_std(mid_l),
            "frac_fine_mean": float(np.nanmean(fine_l)),
            "frac_fine_std": _sample_std(fine_l),
            "band_coarse_f_le": low_freq_max,
            "band_fine_f_ge": high_freq_min,
        })
    return pd.DataFrame(rows)


def plot_figure5c_psd_comparison(
    psd_results: dict[str, dict[str, Any]],
    *,
    figsize: tuple[float, float] = (7.0, 5.0),
    image_size_px: int = 128,
    font_size: float = 14.0,
    font_family: str | None = "Helvetica Neue",
    spine_linewidth: float = 1.5,
    save_path: str | Path | None = None,
    save_dpi: int = 300,
) -> plt.Figure:
    """
    Overlay mean PSD curves (one per model) on shared log-log axes.

    ``psd_results`` maps model label → output of :func:`compute_pc_perturbation_psd`.
    The secondary x-axis shows approximate feature size in pixels (1 / freq).
    """
    colors = {"CNN": "#1f77b4", "FNO": "#ff7f0e", "ViT": "#2ca02c"}
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    for label, res in psd_results.items():
        freq = res["freq"]
        psd = res["mean_psd"]
        valid = freq > 0
        ax.plot(
            freq[valid], psd[valid],
            label=label, color=colors.get(label, None), linewidth=2.0,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Spatial frequency (cycles / pixel)", fontsize=font_size, fontfamily=font_family)
    ax.set_ylabel("Power spectral density", fontsize=font_size, fontfamily=font_family)

    ax.tick_params(axis="both", which="both", labelsize=font_size - 2)
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        if font_family:
            lbl.set_fontfamily(font_family)

    ax2 = ax.secondary_xaxis("top", functions=(lambda f: 1.0 / f, lambda w: 1.0 / w))
    ax2.set_xlabel("Feature size (pixels)", fontsize=font_size, fontfamily=font_family)
    ax2.tick_params(axis="x", labelsize=font_size - 2)
    for lbl in ax2.get_xticklabels():
        if font_family:
            lbl.set_fontfamily(font_family)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(spine_linewidth)

    ax.legend(fontsize=font_size - 1, frameon=False)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=save_dpi, bbox_inches="tight")

    return fig


def show_figure5c_figures(figures: list[plt.Figure]) -> None:
    """Display Figure 5c outputs in Jupyter without extra whitespace (same trick as 5b).

    Closes each figure after display to prevent the inline backend from
    auto-displaying it a second time at the end of the cell.
    """
    if not figures:
        return
    try:
        import io
        from IPython.display import Image, display
        for fig in figures:
            buf = io.BytesIO()
            fig.savefig(
                buf, format="png", dpi=fig.dpi,
                bbox_inches="tight", pad_inches=0.0,
                facecolor="none", edgecolor="none", transparent=True,
            )
            buf.seek(0)
            display(Image(data=buf.getvalue()))
            plt.close(fig)
    except Exception:
        for fig in figures:
            plt.figure(fig.number)
        plt.show()
