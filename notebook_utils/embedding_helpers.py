"""Load ``encoded_data.csv`` from each ``_best_model`` into nested tensors."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from latent_model.loaders.encoded_csv_splits import train_val_test_ids_from_encoded_csv
import torch
import torch.nn as nn


def encoded_csv_path(results_dir: Path, model_name: str, dataset_name: str) -> Path:
    return results_dir / model_name / "_best_model" / dataset_name / "encoded_data.csv"


def sample_ids_by_encoded_csv_split(csv_path: str | Path) -> dict[str, list[str]]:
    """``{\"train\": [...], \"val\": [...], \"test\": [...]}`` from the CSV ``split`` column."""
    tr, va, te = train_val_test_ids_from_encoded_csv(csv_path)
    return {"train": tr, "val": va, "test": te}


def _dim_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("dim_") and c[4:].isdigit()]
    return sorted(cols, key=lambda x: int(x.split("_", 1)[1]))


def read_encoded_split(csv_path: Path, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Latents (N, D) and timepoint labels (N,) for one split in **CSV row order**.
    """
    df = pd.read_csv(csv_path)
    sub = df.loc[df["split"] == split]
    dim_cols = _dim_columns(df)
    arr = sub[dim_cols].to_numpy(dtype="float32")
    z = torch.from_numpy(arr)
    y = torch.tensor(sub["timepoint"].to_numpy(), dtype=torch.long)
    return z, y


def read_encoded_latents(csv_path: Path, split: str) -> torch.Tensor:
    """Latent matrix (N, D) for one split from a single ``encoded_data.csv``."""
    z, _ = read_encoded_split(csv_path, split)
    return z


def read_encoded_metadata(csv_path: Path, split: str) -> tuple[list[int], list[str]]:
    """
    Timepoint and sample_id per row for one split, in **CSV row order** (aligned with latents).
    """
    df = pd.read_csv(csv_path)
    sub = df.loc[df["split"] == split]
    timepoints = [int(x) for x in sub["timepoint"].tolist()]
    sample_ids = [str(x) for x in sub["sample_id"].tolist()]
    return timepoints, sample_ids


def load_encoded_metadata(
    results_dir: str | Path,
    all_models: dict[str, dict[str, Any]],
    split: str = "test",
) -> tuple[dict[str, dict[str, list[int]]], dict[str, dict[str, list[str]]]]:
    """
    Nested timepoints and sample IDs from CSV, same keys and ordering as :func:`load_encoded_latents`.
    """
    rd = Path(results_dir)
    tp_out: dict[str, dict[str, list[int]]] = {}
    sid_out: dict[str, dict[str, list[str]]] = {}

    for model_name, datasets in all_models.items():
        tp_out[model_name] = {}
        sid_out[model_name] = {}
        for dataset_name in datasets:
            path = encoded_csv_path(rd, model_name, dataset_name)
            tps, sids = read_encoded_metadata(path, split)
            tp_out[model_name][dataset_name] = tps
            sid_out[model_name][dataset_name] = sids
    return tp_out, sid_out

def load_encoded_latents(
    results_dir: str | Path,
    all_models: dict[str, dict[str, Any]],
    split: str = "test",
    normalize: bool = False,
) -> dict[str, dict[str, torch.Tensor]]:
    """Nested latents from CSV; pass ``all_models``"""
    rd = Path(results_dir)
    out: dict[str, dict[str, torch.Tensor]] = {}

    for model_name, datasets in all_models.items():
        out[model_name] = {}
        for dataset_name in datasets:
            path = encoded_csv_path(rd, model_name, dataset_name)
            z = read_encoded_latents(path, split)
            if normalize:
                z = (z - z.mean(0, keepdim=True)) / (z.std(0, keepdim=True) + 1e-8)
            out[model_name][dataset_name] = z
    return out

def _timepoint_labels_to_tensor(
    y: torch.Tensor | list[int] | np.ndarray, *, device: torch.device
) -> torch.Tensor:
    """1D ``long`` tensor on ``device`` for CE / accuracy."""
    if isinstance(y, torch.Tensor):
        return y.to(device=device, dtype=torch.long).reshape(-1)
    return torch.as_tensor(np.asarray(y, dtype=np.int64), device=device, dtype=torch.long).reshape(-1)


def _timepoint_labels_to_numpy(y: torch.Tensor | list[int] | np.ndarray) -> np.ndarray:
    if isinstance(y, torch.Tensor):
        return y.detach().cpu().numpy().astype(np.int64, copy=False)
    return np.asarray(y, dtype=np.int64)


def timepoint_decoder_metrics(
    all_models: dict[str, dict[str, nn.Module]],
    latents: dict[str, dict[str, torch.Tensor]],
    true_timepoints: dict[str, dict[str, torch.Tensor | list[int] | np.ndarray]],
    device: torch.device | str = "cpu",
    ) -> dict[str, dict[str, dict[str, float]]]:
    """Mean CE and accuracy for ``decoder_timepoint``"""

    ce = torch.nn.CrossEntropyLoss(reduction="mean")
    out: dict[str, dict[str, dict[str, float]]] = {}

    for model_name, datasets in all_models.items():
        out[model_name] = {}
        for dataset_name, model in datasets.items():
            z = latents[model_name][dataset_name]
            y = true_timepoints[model_name][dataset_name]
            y_t = _timepoint_labels_to_tensor(y, device=device)
            head = model.decoder_timepoint
            head.eval()
            head.to(device)
            try:
                with torch.no_grad():
                    logits = head(z.to(device))
                    loss = ce(logits, y_t).item()
                    acc = (logits.argmax(1) == y_t).float().mean().item()
            finally:
                head.cpu()
            out[model_name][dataset_name] = {"accuracy": acc, "cross_entropy": loss}
    return out


def timepoint_decoder_true_pred(
    all_models: dict[str, dict[str, nn.Module]],
    latents: dict[str, dict[str, torch.Tensor]],
    true_timepoints: dict[str, dict[str, torch.Tensor | list[int] | np.ndarray]],
    device: torch.device | str = "cpu",
    ) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """
    True vs predicted timepoint class index per row.
    ``true_timepoints`` may be CSV lists from :func:`load_encoded_metadata` or tensors.
    """
    if isinstance(device, str):
        device = torch.device(device)
    out: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for model_name, datasets in all_models.items():
        out[model_name] = {}
        for dataset_name, model in datasets.items():
            z = latents[model_name][dataset_name]
            y = true_timepoints[model_name][dataset_name]

            head = model.decoder_timepoint
            head.eval()
            head.to(device)
            with torch.no_grad():
                logits = head(z.to(device))
                pred = logits.argmax(1).cpu().numpy()
            head.cpu()

            out[model_name][dataset_name] = {
                "true": _timepoint_labels_to_numpy(y),
                "pred": pred,
            }
    return out
