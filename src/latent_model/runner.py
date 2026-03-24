import itertools
import json
import os
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from simulation_encoder.logger import Logger

from latent_model.loaders.sequence_loader import SequenceLoader
from latent_model.models.point_models import SupervisedClassifier
from latent_model.models.trajectory_models import LSTMModel, RNNModel, TemporalModel

class TrajectoryModelRunner:
    def __init__(self, logger: Logger | None = None, verbose: bool = False) -> None:
        self.logger = logger
        self.verbose = verbose
        self.models: dict[str, dict[str, list[TemporalModel]]] = {}
        self.loaders: dict[str, dict[str, SequenceLoader]] = {}

    def add_loaders(self, loaders: dict[str, dict[str, SequenceLoader]]) -> None:
        self.loaders = loaders

    def add_models(self, models: dict[str, dict[str, list[TemporalModel]]]) -> None:
        self.models = models

    def get_loader(self, model_name: str, dataset_name: str) -> SequenceLoader:
        return self.loaders[model_name][dataset_name]

    def get_models(self, model_name: str, dataset_name: str) -> list[TemporalModel]:
        return self.models[model_name][dataset_name]

    def run_temporal_model(self) -> dict:
        if not self.loaders:
            raise ValueError("No loaders have been added to the runner.")
        if not self.models:
            raise ValueError("No models have been added to the runner.")

        results = defaultdict(dict)

        for encoder_model_name, dataset_loaders in self.loaders.items():
            for dataset_name, loader in dataset_loaders.items():
                best_model = None
                best_loss = float("inf")

                self._log(
                    f"Finding optimal model for {dataset_name} data encoded with {encoder_model_name}"
                )
                for model in self.models[encoder_model_name][dataset_name]:
                    train_losses, val_losses = self._train_model(model, loader)

                    if val_losses[-1] < best_loss:
                        best_loss = val_losses[-1]
                        best_model = model

                results[encoder_model_name][dataset_name] = {
                    "best_model": best_model,
                    "best_val_loss": best_loss,
                }

        return results

    def _train_model(self, model: TemporalModel, loader: SequenceLoader) -> tuple:
        train_loader = loader.get_dataloader(dataset_type="train")
        val_loader = loader.get_dataloader(dataset_type="val")

        losses, val_losses = model.fit(
            train_loader, val_loader=val_loader, patience=5, min_delta=0.001, max_epochs=1
        )

        return losses, val_losses

    def _eval_model(self, model: TemporalModel, loader: SequenceLoader) -> float:
        model.eval()
        test_loader = loader.get_dataloader(dataset_type="test")
        test_loss = model.eval_one_epoch(test_loader)
        return test_loss

    def _log(self, msg: str, level: str = "info") -> None:
        if self.logger:
            if level == "warning":
                self.logger.warning(msg)
            else:
                self.logger.log(msg)

def gastruloid_binary_label(sample_id: str) -> int:
    """Odd arrays vs even arrays (common gastruloid notebook grouping)."""
    return 1 if sample_id.startswith(("array1", "array3", "array5")) else 0

def discover_models_with_best_checkpoint(study_dir: Path) -> set[str]:
    names: set[str] = set()
    if not study_dir.is_dir():
        return names
    for child in study_dir.iterdir():
        if child.is_dir() and (child / "_best_model").is_dir():
            names.add(child.name)
    return names

def iter_encoded_csv_paths(study_dir: Path) -> Iterator[tuple[str, str, Path]]:
    """Yield (encoder_model_name, dataset_name, encoded_data.csv path)."""
    study_dir = study_dir.resolve()
    for model_path in sorted(study_dir.iterdir()):
        if not model_path.is_dir():
            continue
        best = model_path / "_best_model"
        if not best.is_dir():
            continue
        for ds_path in sorted(best.iterdir()):
            if not ds_path.is_dir():
                continue
            csv_path = ds_path / "encoded_data.csv"
            if csv_path.is_file():
                yield model_path.name, ds_path.name, csv_path

def load_latent_train_val_test(
    csv_path: Path,
    time_point_idx: int,
    label_fn: Callable[[str], int],
    *,
    random_seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Build (X, y) for train/val/test from encoded CSV.
    """
    data = pd.read_csv(csv_path)
    if "sample_id" not in data.columns or "timepoint" not in data.columns:
        raise ValueError(f"{csv_path}: expected columns sample_id and timepoint")

    feature_cols = [c for c in data.columns if c.startswith("dim_")]
    if not feature_cols:
        raise ValueError(f"{csv_path}: no dim_* latent columns found")

    sample_features: dict[str, np.ndarray] = {}
    for sid, grp in data.groupby("sample_id", sort=False):
        grp = grp.sort_values("timepoint")
        arr = grp[feature_cols].values.astype(np.float64)
        if len(arr) <= time_point_idx:
            continue
        sample_features[sid] = arr[time_point_idx]

    ids = list(sample_features.keys())
    if not ids:
        raise ValueError(f"{csv_path}: no samples at time_point_idx={time_point_idx}")

    def pack(id_list: list[str]) -> tuple[pd.DataFrame, pd.Series]:
        X = pd.DataFrame(
            [sample_features[i] for i in id_list],
            index=id_list,
            columns=feature_cols,
        )
        y = pd.Series([label_fn(i) for i in id_list], index=id_list, dtype=int)
        return X, y

    strat = pd.Series({i: label_fn(i) for i in ids}, dtype=int)
    st1 = strat if strat.nunique() > 1 else None
    train_ids, test_ids = train_test_split(
        ids,
        test_size=test_size,
        random_state=random_seed,
        stratify=st1,
    )
    strat_tr = strat.loc[train_ids]
    st2 = strat_tr if strat_tr.nunique() > 1 else None
    train_ids, val_ids = train_test_split(
        train_ids,
        test_size=val_size,
        random_state=random_seed,
        stratify=st2,
    )

    if not train_ids or not test_ids:
        raise ValueError(f"{csv_path}: empty train or test split after partitioning")

    X_tr, y_tr = pack(train_ids)
    X_va, y_va = pack(val_ids) if val_ids else (pd.DataFrame(), pd.Series(dtype=int))
    X_te, y_te = pack(test_ids)
    return X_tr, y_tr, X_va, y_va, X_te, y_te

def run_classifier_on_splits(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    classifier_type: str = "logistic_regression",
    cv_folds: int = 5,
) -> tuple[list[str], list[str], SupervisedClassifier]:
    """Train with grid search on train; refit on train+val; return correct/incorrect test ids."""
    clf = SupervisedClassifier(model_type=classifier_type, eval_metric="accuracy")

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_val_s = (
        pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
        if len(X_val)
        else X_val
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    clf.grid_search(X_train_s, y_train, cv_folds=cv_folds)
    if len(X_val_s):
        X_fit = pd.concat([X_train_s, X_val_s])
        y_fit = pd.concat([y_train, y_val])
    else:
        X_fit, y_fit = X_train_s, y_train
    clf.fit(X_fit, y_fit)

    pred = clf.predict(X_test_s)
    test_ids = list(X_test.index)
    correct: list[str] = []
    incorrect: list[str] = []
    for sid, p, t in zip(test_ids, pred, y_test.loc[test_ids].values):
        (correct if int(p) == int(t) else incorrect).append(sid)

    return correct, incorrect, clf

def run_study_classification(
    study_dir: Path,
    *,
    study_name: str | None = None,
    label_fn: Callable[[str], int] | None = None,
    time_point_idx: int = 0,
    classifier_type: str = "logistic_regression",
    random_seed: int = 42,
    cv_folds: int = 5,
) -> dict[str, Any]:
    """
    For each ``<study_dir>/<encoder>/_best_model/<dataset>/encoded_data.csv``, train a
    classifier and record test-set correct/incorrect sample ids.
    """
    study_dir = study_dir.expanduser().resolve()
    name = study_name or study_dir.name
    label_fn = label_fn or gastruloid_binary_label

    rows: list[dict[str, Any]] = []
    for model_name, dataset_name, csv_path in iter_encoded_csv_paths(study_dir):
        X_tr, y_tr, X_va, y_va, X_te, y_te = load_latent_train_val_test(
            csv_path,
            time_point_idx,
            label_fn,
            random_seed=random_seed,
        )
        correct, incorrect, _ = run_classifier_on_splits(
            X_tr,
            y_tr,
            X_va,
            y_va,
            X_te,
            y_te,
            classifier_type=classifier_type,
            cv_folds=cv_folds,
        )
        rows.append(
            {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "total_correct": len(correct),
                "total_incorrect": len(incorrect),
                "correct_predictions": sorted(correct),
                "incorrect_predictions": sorted(incorrect),
            }
        )

    if not rows:
        raise ValueError(
            f"No encoded_data.csv found under {study_dir}/*/_best_model/*/ (train encoders first)."
        )

    return {
        "study_name": name,
        "model_dataset_results": rows,
        "meta": {
            "results_study_dir": str(study_dir),
            "time_point_idx": time_point_idx,
            "classifier_type": classifier_type,
            "random_seed": random_seed,
            "label": "gastruloid_binary_label" if label_fn is gastruloid_binary_label else None,
        },
    }


def _default_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _trajectory_classifier_test_split(
    model: TemporalModel,
    test_loader,
    label_fn: Callable[[str], int],
    device: torch.device,
) -> tuple[list[str], list[str]]:
    model.eval()
    correct: list[str] = []
    incorrect: list[str] = []
    with torch.no_grad():
        for batch, sample_ids in test_loader:
            x = batch.to(device)
            logits = model.forward(x)
            y_true = model._sequence_classification_targets(sample_ids, label_fn, device)
            if model.output_size == 1:
                pred = (torch.sigmoid(logits).squeeze(-1) >= 0.5).long()
            else:
                pred = logits.argmax(dim=-1)
            for i, sid in enumerate(sample_ids):
                si = str(sid)
                if int(pred[i]) == int(y_true[i]):
                    correct.append(si)
                else:
                    incorrect.append(si)
    return correct, incorrect


def run_study_trajectory_classification(
    study_dir: Path,
    *,
    study_name: str | None = None,
    label_fn: Callable[[str], int] | None = None,
    random_seed: int = 42,
    num_timepoints: int = 9,
    val_split: float = 0.2,
    test_split: float = 0.2,
    cell: str = "lstm",
    num_classes: int = 2,
    max_epochs: int = 50,
    patience: int = 5,
    hidden_sizes: Sequence[int] = (32, 64, 128),
    num_layers_options: Sequence[int] = (1, 2, 3),
) -> dict[str, Any]:
    """
    Classify each sample from its latent trajectory.

    Hyperparameters are selected by validation loss over a grid of ``hidden_sizes`` ×
    ``num_layers_options`` (defaults: 32/64/128 × 1/2/3).
    """
    study_dir = study_dir.expanduser().resolve()
    name = study_name or study_dir.name
    label_fn = label_fn or gastruloid_binary_label
    device = _default_torch_device()
    output_size = num_classes if num_classes > 1 else 1

    rows: list[dict[str, Any]] = []
    for encoder_name, dataset_name, csv_path in iter_encoded_csv_paths(study_dir):
        loader = SequenceLoader(
            str(csv_path),
            max_seq_len=num_timepoints,
            val_split=val_split,
            test_split=test_split,
            random_seed=random_seed,
        )
        n = loader.num_dims
        train_dl = loader.get_dataloader("train")
        val_dl = loader.get_dataloader("val")
        test_dl = loader.get_dataloader("test")

        cls = LSTMModel if cell.lower() == "lstm" else RNNModel

        best_model: TemporalModel | None = None
        best_val = float("inf")
        best_hidden: int | None = None
        best_num_layers: int | None = None
        for hidden_size, num_layers in itertools.product(hidden_sizes, num_layers_options):
            m = cls(
                input_size=n,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size,
                classification=True,
            )
            m.fit_sequence_classification(
                train_dl,
                val_dl,
                label_fn,
                max_epochs=max_epochs,
                patience=patience,
                device=device,
            )
            vloss = m.eval_one_epoch_sequence_classification(val_dl, label_fn, device)
            if vloss < best_val:
                best_val = vloss
                best_model = m
                best_hidden = hidden_size
                best_num_layers = num_layers

        if best_model is None:
            continue

        correct, incorrect = _trajectory_classifier_test_split(best_model, test_dl, label_fn, device)
        rows.append(
            {
                "model_name": encoder_name,
                "dataset_name": dataset_name,
                "best_hidden_size": best_hidden,
                "best_num_layers": best_num_layers,
                "best_val_loss": best_val,
                "total_correct": len(correct),
                "total_incorrect": len(incorrect),
                "correct_predictions": sorted(correct),
                "incorrect_predictions": sorted(incorrect),
            }
        )

    if not rows:
        raise ValueError(
            f"No encoded_data.csv found under {study_dir}/*/_best_model/*/ (train encoders first)."
        )

    return {
        "study_name": name,
        "model_dataset_results": rows,
        "meta": {
            "results_study_dir": str(study_dir),
            "backend": "trajectory",
            "cell": cell.lower(),
            "hidden_sizes": list(hidden_sizes),
            "num_layers_options": list(num_layers_options),
            "num_classes": num_classes,
            "max_epochs": max_epochs,
            "patience": patience,
            "random_seed": random_seed,
            "label": "gastruloid_binary_label" if label_fn is gastruloid_binary_label else None,
        },
    }


def _normalize_per_example_entry(raw: dict[str, Any], index: int) -> dict[str, Any]:
    prefix = f"model_dataset_results[{index}]"
    for key in ("model_name", "dataset_name"):
        if key not in raw or not isinstance(raw[key], str):
            raise ValueError(f"{prefix}: missing or invalid {key!r}")

    correct = raw.get("correct_predictions")
    incorrect = raw.get("incorrect_predictions")
    if not isinstance(correct, list) or not all(isinstance(x, str) for x in correct):
        raise ValueError(f"{prefix}: correct_predictions must be a list of strings")
    if not isinstance(incorrect, list) or not all(isinstance(x, str) for x in incorrect):
        raise ValueError(f"{prefix}: incorrect_predictions must be a list of strings")

    correct_set = set(correct)
    incorrect_set = set(incorrect)
    if correct_set & incorrect_set:
        sid = next(iter(correct_set & incorrect_set))
        raise ValueError(
            f"{prefix} ({raw['model_name']} / {raw['dataset_name']}): "
            f"sample {sid!r} in both correct and incorrect lists"
        )

    tc, ti = raw.get("total_correct"), raw.get("total_incorrect")
    if tc is not None and int(tc) != len(correct_set):
        raise ValueError(
            f"{prefix}: total_correct={tc} but len(correct_predictions)={len(correct_set)}"
        )
    if ti is not None and int(ti) != len(incorrect_set):
        raise ValueError(
            f"{prefix}: total_incorrect={ti} but len(incorrect_predictions)={len(incorrect_set)}"
        )

    per_example: list[dict[str, Any]] = []
    for sid in sorted(correct_set):
        per_example.append({"sample_id": sid, "correct": True})
    for sid in sorted(incorrect_set):
        per_example.append({"sample_id": sid, "correct": False})
    per_example.sort(key=lambda r: r["sample_id"])
    n_total = len(per_example)
    n_correct = len(correct_set)
    out: dict[str, Any] = {
        "model_name": raw["model_name"],
        "dataset_name": raw["dataset_name"],
        "n_total": n_total,
        "n_correct": n_correct,
        "n_incorrect": n_total - n_correct,
        "accuracy": n_correct / n_total if n_total else None,
        "per_example": per_example,
    }
    for k in ("best_hidden_size", "best_num_layers", "best_val_loss"):
        if k in raw:
            out[k] = raw[k]
    return out

def classification_dict_to_per_example(data: dict[str, Any]) -> dict[str, Any]:
    """Expand ``model_dataset_results`` into per-example records (+ summary fields)."""
    mdr = data.get("model_dataset_results")
    if not isinstance(mdr, list):
        raise ValueError("top-level key 'model_dataset_results' must be a list")
    for i, row in enumerate(mdr):
        if not isinstance(row, dict):
            raise ValueError(f"model_dataset_results[{i}] must be an object")
    models = [_normalize_per_example_entry(row, i) for i, row in enumerate(mdr)]
    out: dict[str, Any] = {
        "study_name": data.get("study_name"),
        "n_models": len(models),
        "models": models,
    }
    if "meta" in data:
        out["meta"] = data["meta"]
    return out

def legacy_json_to_per_example(
    classification_json: Path,
    study_dir: Path,
    *,
    output_json: Path | None = None,
    indent: int = 2,
) -> Path:
    """
    Read an existing ``*_classification_results.json``, keep only models that have
    ``_best_model`` under ``study_dir``, write ``*_per_example.json``.
    """
    classification_json = classification_json.expanduser().resolve()
    study_dir = study_dir.expanduser().resolve()
    allowed = discover_models_with_best_checkpoint(study_dir)
    if not allowed:
        raise FileNotFoundError(f"No _best_model subdirs under {study_dir}")

    with open(classification_json, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("root JSON must be an object")

    mdr = [r for r in data.get("model_dataset_results", []) if r.get("model_name") in allowed]
    if not mdr:
        raise ValueError("No model_dataset_results rows match checkpoint subfolders")
    filtered = {**data, "model_dataset_results": mdr}
    result = classification_dict_to_per_example(filtered)
    result["source_file"] = str(classification_json)
    result["results_study_dir"] = str(study_dir)
    result["models_included"] = sorted(allowed)

    out = output_json or classification_json.with_name(
        f"{classification_json.stem}_per_example.json"
    )
    out = out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    ind = None if indent == 0 else indent
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=ind)
        if ind:
            f.write("\n")
    return out

def build_trajectory_loaders(
    results_path: str | Path,
    seq_len: int | None = None,
    *,
    val_split: float = 0.2,
    test_split: float = 0.2,
    random_seed: int = 42,
) -> dict[str, dict[str, SequenceLoader]]:
    """One :class:`~latent_model.loaders.sequence_loader.SequenceLoader` per ``_best_model`` dataset.

    Train/val/test are always redrawn randomly over all samples (CSV ``split`` column is ignored).
    """
    results_path = os.fspath(results_path)
    loaders: dict[str, dict[str, SequenceLoader]] = {}
    for model_name in os.listdir(results_path):
        loaders[model_name] = {}
        model_path = f"{results_path}/{model_name}/_best_model"
        if not os.path.isdir(model_path):
            continue
        for dataset_name in os.listdir(model_path):
            csv_path = f"{model_path}/{dataset_name}/encoded_data.csv"
            if not os.path.isfile(csv_path):
                continue
            loaders[model_name][dataset_name] = SequenceLoader(
                csv_path,
                max_seq_len=seq_len,
                val_split=val_split,
                test_split=test_split,
                random_seed=random_seed,
            )
    return loaders

def temporal_model_sweep(
    input_size: int,
    output_size: int,
    *,
    classification: bool = False,
    hidden_sizes: tuple[int, ...] = (32, 64, 128),
    num_layers_opts: tuple[int, ...] = (1, 2, 3),
) -> list[TemporalModel]:
    """Cartesian product of RNN/LSTM × hidden size × depth."""
    models: list[TemporalModel] = []
    for model_class, hidden_size, num_layer in itertools.product(
        (RNNModel, LSTMModel), hidden_sizes, num_layers_opts
    ):
        models.append(
            model_class(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layer,
                output_size=output_size,
                classification=classification,
            )
        )
    return models

def build_trajectory_models_for_study(
    results_path: str | Path,
    loaders: dict[str, dict[str, SequenceLoader]],
    *,
    classification: bool = False,
) -> dict[str, dict[str, list[TemporalModel]]]:
    """Build a model candidate list per (encoder, dataset) using each loader's ``num_dims``."""
    results_path = os.fspath(results_path)
    temporal_models: dict[str, dict[str, list[TemporalModel]]] = {}
    for model_name in os.listdir(results_path):
        temporal_models[model_name] = {}
        model_path = f"{results_path}/{model_name}/_best_model"
        if not os.path.isdir(model_path):
            continue
        for dataset_name in os.listdir(model_path):
            if model_name not in loaders or dataset_name not in loaders[model_name]:
                continue
            loader = loaders[model_name][dataset_name]
            n = loader.num_dims
            temporal_models[model_name][dataset_name] = temporal_model_sweep(
                n, n, classification=classification
            )
    return temporal_models
