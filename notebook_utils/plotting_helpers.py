"""Plotting helpers for analysis and manuscript notebooks."""

import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from matplotlib.colors import hex2color


def plot_image_grid(images_data, fig, axes):
    for item in images_data:
        row = item["row"]
        col = item["col"]
        image = item["image"]
        title = item.get("title", "")
        cmap = item.get("cmap", "gray")

        if axes.ndim == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]

        if title:
            ax.set_title(title, pad=0, fontsize=14)

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        imshow_kw = {"cmap": cmap}
        if "vmin" in item and "vmax" in item:
            imshow_kw["vmin"] = item["vmin"]
            imshow_kw["vmax"] = item["vmax"]
        ax.imshow(image, **imshow_kw)
        ax.axis("off")


def get_original_images(samples, labels, channels, start_row, title="Original Timepoint"):
    images_data = []
    for j, sample in enumerate(samples):
        images = sample.squeeze()

        if images.ndim == 2:
            images_data.append(
                {"image": images, "title": f"{title} {labels[j]}", "row": start_row, "col": j}
            )
        elif images.ndim == 3:
            for i, image in enumerate(images):
                t = f"{title} {labels[j]}" if i == 0 else ""
                images_data.append({"image": image, "title": t, "row": start_row + i, "col": j})
        else:
            raise ValueError("Unsupported image dimensions. Expected 2D or 3D images.")

    return images_data


def get_reconstructed_images(samples, model, channels, start_row, title="Predicted Timepoint"):
    images_data = []
    for j, sample in enumerate(samples):
        model.eval()
        sample_input = sample.unsqueeze(0)

        with torch.no_grad():
            result = model(sample_input)

        reconstructed_image, label_pred = result[:2]
        reconstructed_image = reconstructed_image.squeeze()
        label_pred = torch.max(label_pred, dim=1)[1].item()

        if reconstructed_image.ndim == 2:
            images_data.append(
                {"image": reconstructed_image, "title": f"{title}", "row": start_row, "col": j}
            )
        elif reconstructed_image.ndim == 3:
            for i, image in enumerate(reconstructed_image):
                t = f"{title} {label_pred}" if i == 0 else ""
                images_data.append({"image": image, "title": t, "row": start_row + i, "col": j})
        else:
            raise ValueError("Unsupported image dimensions. Expected 2D or 3D images.")

    return images_data


def get_difference_images(samples, model, channels, start_row, cmap="bwr"):
    imgs = []
    for j, sample in enumerate(samples):
        with torch.no_grad():
            recon, _ = model(sample.unsqueeze(0))[:2]
        recon, orig = recon.squeeze(), sample.squeeze()

        diffs = recon - orig
        if diffs.ndim == 2:
            diffs = diffs.unsqueeze(0)

        for ch_idx, diff in enumerate(diffs):
            m = diff.abs().max().item() or 1.0
            imgs.append(
                {
                    "image": diff,
                    "title": "",
                    "row": start_row + ch_idx,
                    "col": j,
                    "cmap": cmap,
                    "vmin": -m,
                    "vmax": m,
                }
            )
    return imgs


def get_saliency_maps(dataset, indices, model, start_row, title="Saliency Map"):
    images_data = []
    for j, idx in enumerate(indices):
        data = dataset[idx]
        input_tensor = data[0].unsqueeze(0)

        saliency_map = model.get_saliency_map(input_tensor).squeeze()

        images_data.append(
            {"image": saliency_map, "title": f"{title}", "row": start_row, "col": j, "cmap": "hot"}
        )

    return images_data


def get_shaded_color(base_hex: str, timepoint: float, min_tp: float, max_tp: float, light_factor: float = 0.35, richness_curve: float = 3.0) -> list[float]:
    base_rgb = hex2color(base_hex)
    if max_tp == min_tp:
        normalized_tp = 1.0
    else:
        normalized_tp = (timepoint - min_tp) / (max_tp - min_tp)
    normalized_tp = max(0.0, min(1.0, float(normalized_tp)))
    u = 1.0 - (1.0 - normalized_tp) ** richness_curve
    interpolation_amount = light_factor + u * (1.0 - light_factor)

    shaded_rgb = [1.0 * (1.0 - interpolation_amount) + c * interpolation_amount for c in base_rgb]
    shaded_rgb = [max(0.0, min(1.0, c)) for c in shaded_rgb]
    return shaded_rgb


# Parity 2c: cell counts + size legend

PARITY_BASE_DOT = 36.0
PARITY_COUNT_POWER = 1.0
PARITY_FACE = "#83ABCF"
PARITY_EDGE = "#6F96B8"
PARITY_ALPHA = 0.9


def parity_cell_counts(y_true, y_pred) -> np.ndarray:
    """Counts per unique (true, pred) pair."""
    pairs = np.stack(
        [np.asarray(y_true, dtype=np.int64).ravel(), np.asarray(y_pred, dtype=np.int64).ravel()],
        axis=1,
    )
    _, counts = np.unique(pairs, axis=0, return_counts=True)
    return counts


def plot_parity_size_legend(
    legend_counts: list[int],
    *,
    base_dot_size: float = PARITY_BASE_DOT,
    count_size_power: float = PARITY_COUNT_POWER,
    face: str = PARITY_FACE,
    edge: str = PARITY_EDGE,
    alpha: float = PARITY_ALPHA,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Legend figure: marker size ~ sample count per (true, pred) cell."""
    fig, ax = plt.subplots(figsize=(3.0, 5.0))
    ax.set_axis_off()
    for n in legend_counts:
        ax.scatter(
            [],
            [],
            s=base_dot_size * (float(n) ** count_size_power),
            c=face,
            alpha=alpha,
            edgecolors=edge,
            linewidths=1.0,
            label=f"{n} samples",
        )
    ax.legend(
        scatterpoints=1,
        frameon=True,
        loc="center",
        title="Samples per (true, pred) cell",
        handletextpad=1.2,
        borderaxespad=0.5,
    )
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
    return fig


# Downstream emergent-property parity (classification_results JSON)

EMERGENT_PARITY_GRAY = "0.42"
EMERGENT_PARITY_DIAG = "0.35"
EMERGENT_PARITY_SCATTER_SIZE = 42
EMERGENT_PARITY_SCATTER_ALPHA = 0.22


def load_point_regression_json(results_dir: Path, emergent_property: str) -> dict:
    """Load first JSON under ``results_dir`` whose meta emergent_property_name matches."""
    prop = emergent_property.strip()
    for path in sorted(Path(results_dir).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("meta") or {}
        name = meta.get("emergent_property_name")
        if str(name).strip().upper() == prop.upper():
            return data
    raise FileNotFoundError(f"No JSON matching emergent property {prop!r} in {results_dir}")


def best_row_per_encoder(rows: list[dict]) -> dict[str, dict]:
    """One row per model_name, highest test_r2."""
    by_model: dict[str, dict] = {}
    for row in rows:
        mname = row["model_name"]
        r2 = float(row["test_r2"])
        prev = by_model.get(mname)
        if prev is None or r2 > float(prev["test_r2"]):
            by_model[mname] = row
    return by_model


def _emergent_y_yhat(row: dict) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(row["test_y_true"], dtype=float).ravel()
    yhat = np.asarray(row["test_y_pred"], dtype=float).ravel()
    return y, yhat


def _emergent_equal_aspect_datalim_silent(ax: plt.Axes) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Ignoring fixed [xy] limits to fulfill fixed data aspect with adjustable data limits\.",
            category=UserWarning,
        )
        ax.set_aspect("equal", adjustable="datalim")


def _emergent_style_axes(
    ax: plt.Axes, tick_labelsize: float, show_ticks: bool = True
) -> None:
    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if show_ticks:
        ax.tick_params(axis="both", which="both", length=3, width=0.6, labelsize=tick_labelsize)
    else:
        ax.tick_params(
            axis="both",
            which="both",
            length=0,
            width=0,
            labelleft=False,
            labelbottom=False,
            left=False,
            bottom=False,
        )


def _emergent_tick_formatters(ax: plt.Axes, xticks: np.ndarray, yticks: np.ndarray) -> None:
    def scale(vals: np.ndarray) -> float:
        v = np.asarray(vals, dtype=float).ravel()
        m = float(np.max(np.abs(v)))
        return 1.0 if m == 0 else float(10 ** np.floor(np.log10(m)))

    sx, sy = scale(xticks), scale(yticks)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _p: f"{x / sx:.2g}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _p: f"{x / sy:.2g}"))
    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)


def emergent_parity_tick_limits(
    prop: str,
    row_ticks: dict[str, dict[str, list[float] | None]] | None,
    xa: float,
    xb: float,
    ya: float,
    yb: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    cfg = (row_ticks or {}).get(prop, {})
    x_user, y_user = cfg.get("x"), cfg.get("y")
    xu = np.asarray(x_user, dtype=float).ravel() if x_user is not None else np.array([])
    yu = np.asarray(y_user, dtype=float).ravel() if y_user is not None else np.array([])

    if xu.size:
        xa = min(xa, float(xu.min()))
        xb = max(xb, float(xu.max()))
        xticks = xu
    else:
        xticks = np.linspace(xa, xb, 3)

    if yu.size:
        ya = min(ya, float(yu.min()))
        yb = max(yb, float(yu.max()))
        yticks = yu
    else:
        yticks = np.linspace(ya, yb, 3)

    return xticks, yticks, xa, xb, ya, yb


def plot_emergent_parity_row(
    emergent_property: str,
    payload: dict,
    models: list[str],
    fig_width_per_col: float = 2.15,
    fig_row_height: float = 2.05,
    row_ticks: dict[str, dict[str, list[float] | None]] | None = None,
    tick_labelsize: float = 12,
    model_panels: Literal["horizontal", "vertical"] = "horizontal",
    show_ticks: bool = True,
) -> plt.Figure:
    """
    Parity scatters: one panel per encoder in ``models`` order.
    """
    prop = emergent_property
    n = len(models)
    if model_panels == "horizontal":
        fig_w, fig_h = fig_width_per_col * n, fig_row_height
        fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), squeeze=False, constrained_layout=False)
        panel_axes = list(axes[0])
    else:
        fig_w, fig_h = fig_width_per_col, fig_row_height * n
        fig, axes = plt.subplots(n, 1, figsize=(fig_w, fig_h), squeeze=False, constrained_layout=False)
        panel_axes = np.ravel(axes).tolist()

    best = best_row_per_encoder(payload["model_dataset_results"])
    y_lo_r, y_hi_r = np.inf, -np.inf
    x_lo_r, x_hi_r = np.inf, -np.inf
    for model_name in models:
        row = best.get(model_name)
        if row is None:
            continue
        y, yhat = _emergent_y_yhat(row)
        y_lo_r = min(y_lo_r, float(np.nanmin(yhat)))
        y_hi_r = max(y_hi_r, float(np.nanmax(yhat)))
        x_lo_r = min(x_lo_r, float(np.nanmin(y)))
        x_hi_r = max(x_hi_r, float(np.nanmax(y)))

    if not np.isfinite(y_lo_r):
        for ax in panel_axes:
            ax.set_axis_off()
        if model_panels == "horizontal":
            fig.subplots_adjust(left=0.08, right=0.99, top=0.99, bottom=0.08, wspace=0.12)
        else:
            fig.subplots_adjust(left=0.08, right=0.99, top=0.99, bottom=0.08, hspace=0.12)
        return fig

    span_y = y_hi_r - y_lo_r
    pad_y = 0.05 * span_y if span_y > 0 else 1.0
    ya, yb = y_lo_r - pad_y, y_hi_r + pad_y
    span_x = x_hi_r - x_lo_r
    pad_x = 0.05 * span_x if span_x > 0 else 1.0
    xa, xb = x_lo_r - pad_x, x_hi_r + pad_x

    xticks, yticks, xa, xb, ya, yb = emergent_parity_tick_limits(prop, row_ticks, xa, xb, ya, yb)

    for ci, model_name in enumerate(models):
        ax = panel_axes[ci]
        _emergent_style_axes(ax, tick_labelsize=tick_labelsize, show_ticks=show_ticks)
        row = best.get(model_name)
        if row is None:
            ax.set_axis_off()
            continue
        y, yhat = _emergent_y_yhat(row)

        ax.scatter(
            y,
            yhat,
            s=EMERGENT_PARITY_SCATTER_SIZE,
            alpha=EMERGENT_PARITY_SCATTER_ALPHA,
            c="#152F67",
            edgecolors="none",
            rasterized=True,
        )
        ax.set_xlim(xa, xb)
        ax.set_ylim(ya, yb)
        if show_ticks:
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.minorticks_off()
        _emergent_equal_aspect_datalim_silent(ax)
        xl, yl = ax.get_xlim(), ax.get_ylim()
        t0, t1 = max(xl[0], yl[0]), min(xl[1], yl[1])
        if t1 >= t0:
            ax.plot([t0, t1], [t0, t1], color=EMERGENT_PARITY_DIAG, linestyle="--", linewidth=0.9)
        if show_ticks:
            if model_panels == "horizontal":
                ax.tick_params(
                    labelleft=ci == 0,
                    labelbottom=True,
                    left=ci == 0,
                    bottom=True,
                    labelsize=tick_labelsize,
                )
            else:
                ax.tick_params(
                    labelleft=ci == 0,
                    labelbottom=ci == n - 1,
                    left=ci == 0,
                    bottom=ci == n - 1,
                    labelsize=tick_labelsize,
                )
            _emergent_tick_formatters(ax, xticks, yticks)
            for tick in (*ax.get_xticklabels(), *ax.get_yticklabels()):
                tick.set_fontsize(tick_labelsize)

    if model_panels == "horizontal":
        fig.subplots_adjust(left=0.08, right=0.99, top=0.99, bottom=0.08, wspace=0.12)
    else:
        fig.subplots_adjust(left=0.08, right=0.99, top=0.99, bottom=0.08, hspace=0.12)
    return fig


def print_emergent_parity_legend(
    emergent_properties: list[str],
    payloads: list[dict],
    model_order: list[str],
    model_panels: Literal["horizontal", "vertical"] = "horizontal",
    ) -> None:
    """Print row order and R² table (stdout)."""
    print("Emergent properties (top → bottom row order):")
    for i, name in enumerate(emergent_properties, start=1):
        print(f"  {i}. {name}")
    w_prop = max(len("property"), max((len(p) for p in emergent_properties), default=8))
    w_model = max(10, max((len(m) for m in model_order), default=8))
    axis_blurb = (
        "columns left → right"
        if model_panels == "horizontal"
        else "rows top → bottom"
    )
    print(f"\nTest R² (best row per encoder; panels {axis_blurb}):")
    hdr = f"  {'property':<{w_prop}}" + "".join(f"{m:>{w_model}}" for m in model_order)
    print(hdr)
    print("  " + "-" * (w_prop + w_model * len(model_order)))
    for prop, payload in zip(emergent_properties, payloads):
        best = best_row_per_encoder(payload["model_dataset_results"])
        parts = [f"  {prop:<{w_prop}}"]
        for m in model_order:
            row = best.get(m)
            parts.append(
                f"{float(row['test_r2']):>{w_model}.3f}" if row is not None else f"{'—':>{w_model}}"
            )
        print("".join(parts))


# --- Figure 3: PCA trajectory panels by confusion bucket --------------------

def plot_trajectory_confusion_panels(
    emb: np.ndarray,
    ids: list | np.ndarray,
    tps: list | np.ndarray,
    incorrect: set,
    y_true_by_sid: dict[Any, int],
    *,
    class_labels: tuple[str, str],
    fg_top_color: str,
    fg_bottom_color: str,
    log_prefix: str | None = None,
    bg_color: str = "#D9D9D9",
    bg_alpha: float = 0.18,
    fg_alpha: float = 0.85,
    fg_linewidth: float = 2.0,
    fg_point_size: float = 25,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """2×2 PCA panels: background trajectories + highlighted bucket."""
    y_pred_by_sid = {s: (yt if s not in incorrect else 1 - yt) for s, yt in y_true_by_sid.items()}
    buckets: dict[tuple[int, int], set] = {(0, 0): set(), (0, 1): set(), (1, 0): set(), (1, 1): set()}
    for s in y_true_by_sid:
        buckets[(y_true_by_sid[s], y_pred_by_sid[s])].add(s)

    traj_all: dict[Any, list] = defaultdict(list)
    emb_all_list = []
    for i, (sid, tp) in enumerate(zip(ids, tps)):
        traj_all[sid].append((tp, emb[i]))
        emb_all_list.append(emb[i])
    emb_all = np.stack(emb_all_list, axis=0)
    tmin, tmax = float(min(tps)), float(max(tps))

    panel_specs = [
        ((0, 0), f"{class_labels[0]} → {class_labels[0]}"),
        ((0, 1), f"{class_labels[0]} → {class_labels[1]}"),
        ((1, 0), f"{class_labels[1]} → {class_labels[0]}"),
        ((1, 1), f"{class_labels[1]} → {class_labels[1]}"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()

    for ax, ((yt, yp), title) in zip(axes, panel_specs):
        sids_fg = buckets[(yt, yp)]
        if log_prefix is not None:
            print(f"  {log_prefix} | {title} | n={len(sids_fg)}")
        ax.scatter(
            emb_all[:, 0],
            emb_all[:, 1],
            c=bg_color,
            s=fg_point_size,
            edgecolor="none",
            alpha=bg_alpha,
        )
        for sid, entries in traj_all.items():
            entries.sort(key=lambda x: x[0])
            for (t1, e1), (_, e2) in zip(entries, entries[1:]):
                ax.plot(
                    [e1[0], e2[0]],
                    [e1[1], e2[1]],
                    color=bg_color,
                    linestyle="-",
                    linewidth=1.5,
                    alpha=bg_alpha,
                )

        row_fg = fg_top_color if yt == 0 else fg_bottom_color
        for sid in sorted(sids_fg):
            entries = traj_all.get(sid) or []
            entries.sort(key=lambda x: x[0])
            if not entries:
                continue
            for (t1, e1), (_, e2) in zip(entries, entries[1:]):
                cseg = get_shaded_color(row_fg, t1, tmin, tmax)
                ax.plot(
                    [e1[0], e2[0]],
                    [e1[1], e2[1]],
                    color=cseg,
                    linestyle="--",
                    linewidth=fg_linewidth,
                    alpha=fg_alpha,
                )
            pts = np.stack([e for (_, e) in entries], axis=0)
            cols = [get_shaded_color(row_fg, tp, tmin, tmax) for tp, _ in entries]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=cols,
                s=fg_point_size,
                edgecolor="none",
                alpha=fg_alpha,
            )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout(pad=0.5, rect=[0.0, 0.0, 1.0, 0.96])
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0)
    return fig
