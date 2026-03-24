"""Load ``encoded_data.csv`` from each ``_best_model`` run into nested tensors."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def encoded_csv_path(results_dir: Path, model_name: str, dataset_name: str) -> Path:
    return results_dir / model_name / "_best_model" / dataset_name / "encoded_data.csv"


def _dim_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("dim_") and c[4:].isdigit()]
    return sorted(cols, key=lambda x: int(x.split("_", 1)[1]))


def read_encoded_split(csv_path: Path, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Latents (N, D) and timepoint labels (N,) for one split in **CSV row order**.
    """
    df = pd.read_csv(csv_path)
    if "split" not in df.columns:
        raise ValueError(f"{csv_path}: missing 'split' column")
    if "timepoint" not in df.columns:
        raise ValueError(f"{csv_path}: missing 'timepoint' column")
    sub = df.loc[df["split"] == split]
    dim_cols = _dim_columns(df)
    if not dim_cols:
        raise ValueError(f"{csv_path}: no dim_* columns")
    if sub.empty:
        raise ValueError(f"{csv_path}: no rows for split={split!r}")
    arr = sub[dim_cols].to_numpy(dtype="float32")
    z = torch.from_numpy(arr)
    y = torch.tensor(sub["timepoint"].to_numpy(), dtype=torch.long)
    if z.shape[0] != y.shape[0]:
        raise ValueError(f"{csv_path}: latent rows {z.shape[0]} != timepoint rows {y.shape[0]}")
    return z, y


def read_encoded_latents(csv_path: Path, split: str) -> torch.Tensor:
    """Latent matrix (N, D) for one split from a single ``encoded_data.csv``."""
    z, _ = read_encoded_split(csv_path, split)
    return z


def load_encoded_metadata(csv_path: Path, split: str) -> tuple[list[int], list[str]]:
    """
    Timepoint and sample_id per row for one split, in **CSV row order** (aligned with latents).
    """
    df = pd.read_csv(csv_path)
    sub = df.loc[df["split"] == split]
    if sub.empty:
        raise ValueError(f"{csv_path}: no rows for split={split!r}")
    timepoints = [int(x) for x in sub["timepoint"].tolist()]
    sample_ids = [str(x) for x in sub["sample_id"].tolist()]
    if len(timepoints) != len(sample_ids):
        raise ValueError(
            f"{csv_path}: timepoint rows {len(timepoints)} != sample_id rows {len(sample_ids)}"
        )
    return timepoints, sample_ids


def load_encoded_metadata(
    results_dir: str | Path,
    all_models: dict[str, dict[str, Any]] | None = None,
    split: str = "test",
) -> tuple[dict[str, dict[str, list[int]]], dict[str, dict[str, list[str]]]]:
    """
    Nested timepoints and sample IDs from CSV, same keys and ordering as :func:`load_encoded_latents`.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(split)
    rd = Path(results_dir)
    tp_out: dict[str, dict[str, list[int]]] = {}
    sid_out: dict[str, dict[str, list[str]]] = {}

    if all_models is not None:
        for model_name, datasets in all_models.items():
            tp_out[model_name] = {}
            sid_out[model_name] = {}
            for dataset_name in datasets:
                path = encoded_csv_path(rd, model_name, dataset_name)
                tps, sids = read_encoded_metadata(path, split)
                tp_out[model_name][dataset_name] = tps
                sid_out[model_name][dataset_name] = sids
        return tp_out, sid_out

    for model_path in sorted(p for p in rd.iterdir() if p.is_dir()):
        best = model_path / "_best_model"
        if not best.is_dir():
            continue
        mn = model_path.name
        tp_out[mn] = {}
        sid_out[mn] = {}
        for ds_path in sorted(p for p in best.iterdir() if p.is_dir()):
            csv_path = ds_path / "encoded_data.csv"
            if not csv_path.is_file():
                continue
            tps, sids = read_encoded_metadata(csv_path, split)
            tp_out[mn][ds_path.name] = tps
            sid_out[mn][ds_path.name] = sids
    return {k: v for k, v in tp_out.items() if v}, {k: v for k, v in sid_out.items() if v}


def load_encoded_latents(
    results_dir: str | Path,
    all_models: dict[str, dict[str, Any]] | None = None,
    split: str = "test",
    normalize: bool = False,
) -> dict[str, dict[str, torch.Tensor]]:
    """Nested latents from CSV; pass ``all_models`` to restrict keys (matches ``load_models``)."""
    if split not in ("train", "val", "test"):
        raise ValueError(split)
    rd = Path(results_dir)
    out: dict[str, dict[str, torch.Tensor]] = {}

    if all_models is not None:
        for model_name, datasets in all_models.items():
            out[model_name] = {}
            for dataset_name in datasets:
                path = encoded_csv_path(rd, model_name, dataset_name)
                z = read_encoded_latents(path, split)
                if normalize:
                    z = (z - z.mean(0, keepdim=True)) / (z.std(0, keepdim=True) + 1e-8)
                out[model_name][dataset_name] = z
        return out

    for model_path in sorted(p for p in rd.iterdir() if p.is_dir()):
        best = model_path / "_best_model"
        if not best.is_dir():
            continue
        mn = model_path.name
        out[mn] = {}
        for ds_path in sorted(p for p in best.iterdir() if p.is_dir()):
            csv_path = ds_path / "encoded_data.csv"
            if not csv_path.is_file():
                continue
            z = read_encoded_latents(csv_path, split)
            if normalize:
                z = (z - z.mean(0, keepdim=True)) / (z.std(0, keepdim=True) + 1e-8)
            out[mn][ds_path.name] = z
    return {k: v for k, v in out.items() if v}


def load_encoded_latents_and_timepoints(
    results_dir: str | Path,
    all_models: dict[str, dict[str, Any]] | None = None,
    split: str = "test",
    normalize: bool = False,
    ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, torch.Tensor]]]:
    results_path = Path(results_dir)
    z_out: dict[str, dict[str, torch.Tensor]] = {}
    y_out: dict[str, dict[str, torch.Tensor]] = {}

    if all_models is not None:
        for model_name, datasets in all_models.items():
            z_out[model_name] = {}
            y_out[model_name] = {}
            for dataset_name in datasets:
                path = encoded_csv_path(results_path, model_name, dataset_name)
                z, y = read_encoded_split(path, split)
                if normalize:
                    z = (z - z.mean(0, keepdim=True)) / (z.std(0, keepdim=True) + 1e-8)
                z_out[model_name][dataset_name] = z
                y_out[model_name][dataset_name] = y
        return z_out, y_out

    for model_path in sorted(p for p in results_path.iterdir() if p.is_dir()):
        best = model_path / "_best_model"
        if not best.is_dir():
            continue
        mn = model_path.name
        z_out[mn] = {}
        y_out[mn] = {}
        for ds_path in sorted(p for p in best.iterdir() if p.is_dir()):
            csv_path = ds_path / "encoded_data.csv"
            if not csv_path.is_file():
                continue
            z, y = read_encoded_split(csv_path, split)
            if normalize:
                z = (z - z.mean(0, keepdim=True)) / (z.std(0, keepdim=True) + 1e-8)
            z_out[mn][ds_path.name] = z
            y_out[mn][ds_path.name] = y
    return {k: v for k, v in z_out.items() if v}, {k: v for k, v in y_out.items() if v}


def timepoint_decoder_metrics(
    all_models: dict[str, dict[str, nn.Module]],
    latents: dict[str, dict[str, torch.Tensor]],
    true_timepoints: dict[str, dict[str, torch.Tensor]],
    device: torch.device | str = "cpu",
) -> dict[str, dict[str, dict[str, float]]]:
    """Mean CE and accuracy for ``decoder_timepoint`` using CSV-aligned latents and labels."""
    if isinstance(device, str):
        device = torch.device(device)
    ce = torch.nn.CrossEntropyLoss(reduction="mean")
    out: dict[str, dict[str, dict[str, float]]] = {}

    for model_name, datasets in all_models.items():
        out[model_name] = {}
        for dataset_name, model in datasets.items():
            z = latents[model_name][dataset_name]
            y = true_timepoints[model_name][dataset_name]
            if z.shape[0] != y.shape[0]:
                raise ValueError(
                    f"{model_name}/{dataset_name}: latent rows {z.shape[0]} != label rows {y.shape[0]}"
                )

            head = model.decoder_timepoint
            head.eval()
            head.to(device)
            try:
                with torch.no_grad():
                    logits = head(z.to(device))
                    loss = ce(logits, y.to(device)).item()
                    acc = (logits.argmax(1).cpu() == y).float().mean().item()
            finally:
                head.cpu()
            out[model_name][dataset_name] = {"accuracy": acc, "cross_entropy": loss}
    return out


def timepoint_decoder_true_pred(
    all_models: dict[str, dict[str, nn.Module]],
    latents: dict[str, dict[str, torch.Tensor]],
    true_timepoints: dict[str, dict[str, torch.Tensor]],
    device: torch.device | str = "cpu",
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """
    True vs predicted timepoint class index per row
    """
    if isinstance(device, str):
        device = torch.device(device)
    out: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for model_name, datasets in all_models.items():
        out[model_name] = {}
        for dataset_name, model in datasets.items():
            z = latents[model_name][dataset_name]
            y = true_timepoints[model_name][dataset_name]
            if z.shape[0] != y.shape[0]:
                raise ValueError(
                    f"{model_name}/{dataset_name}: latent rows {z.shape[0]} != label rows {y.shape[0]}"
                )

            head = model.decoder_timepoint
            head.eval()
            head.to(device)
            try:
                with torch.no_grad():
                    logits = head(z.to(device))
                    pred = logits.argmax(1).cpu().numpy()
            finally:
                head.cpu()

            out[model_name][dataset_name] = {
                "true": y.numpy(),
                "pred": pred,
            }
    return out
