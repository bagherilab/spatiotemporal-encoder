"""Paper figures: data + matplotlib helpers"""

import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from notebook_utils.embedding_helpers import (
    load_encoded_latents_and_timepoints,
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
        linewidth=1.2,
    )
    ax.bar(
        bar_group_x + bar_width / 2,
        test_mse_per_group,
        width=bar_width,
        facecolor="#808080",
        edgecolor="black",
        linewidth=0.8,
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
    spine_linewidth: float = 1.0,
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
        alpha=0.9,
        linewidth=1.0,
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
    for side in ("left", "bottom", "top", "right"):
        sp = ax.spines[side]
        sp.set_linewidth(spine_linewidth)
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig, ax

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
    normalize: bool = False,
    ) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """
    True vs predicted timepoint class index per model/dataset.
    """
    latents, true_tp = load_encoded_latents_and_timepoints(
        results_dir, all_models=all_models, split=split, normalize=normalize
    )
    return timepoint_decoder_true_pred(all_models, latents, true_tp, device=device)

class _CpuBatchDataLoader:
    """Iterates like a DataLoader but moves all tensors in each batch to CPU."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader

    def __iter__(self) -> Iterator[Any]:
        for batch in self._loader:
            yield _tensors_to_cpu(batch)

    def __len__(self) -> int:
        return len(self._loader)
