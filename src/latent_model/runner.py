import itertools
import os
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from latent_model.loaders.encoded_csv_splits import train_val_test_ids_from_encoded_csv
from latent_model.loaders.sequence_loader import SequenceLoader
from latent_model.models.point_models import SupervisedClassifier, SupervisedRegressor
from latent_model.models.trajectory_models import LSTMModel, RNNModel, TemporalModel


class TrajectoryModelRunner:
    def __init__(self, verbose: bool = False) -> None:
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


def gastruloid_binary_label(sample_id: str) -> int:
    """
    0 — euploid: ``array1``      1 — aneuploid: ``array2``
    """
    sid = str(sample_id)
    if sid.startswith("array1"):
        return 0
    if sid.startswith("array2"):
        return 1
    raise ValueError(f"Unknown gastruloid sample_id (expected array1–6): {sample_id!r}")


def arcade_colony_tissue_label(sample_id: str) -> int:
    """
    0 — colony: ``C_*``       1 — CH context: ``CH_*``
    """
    sid = str(sample_id)
    if sid.startswith("CH_"):
        return 1
    if sid.startswith("C_"):
        return 0
    raise ValueError(f"Unknown ARCADE sample_id (expected C_* or CH*): {sample_id!r}")


def _label_meta_name(label_fn: Callable[[str], int]) -> str:
    return getattr(label_fn, "__name__", repr(label_fn))


def iter_encoded_csv_paths(study_dir: Path) -> Iterator[tuple[str, str, Path]]:
    """Yield (encoder_model_name, dataset_name, encoded_data.csv path)."""
    study_dir = study_dir.resolve()
    for model_path in sorted(study_dir.iterdir()):
        best = model_path / "_best_model"
        for ds_path in sorted(best.iterdir()):
            csv_path = ds_path / "encoded_data.csv"
            if csv_path.is_file():
                yield model_path.name, ds_path.name, csv_path


def load_latent_train_val_test(
    csv_path: Path,
    time_point_idx: int,
    label_fn: Callable[[str], int],
    random_seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.2,
    split_mode: str = "csv",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Build (X, y) for train/val/test from encoded CSV.

    ``split_mode=\"csv\"`` uses the CSV ``split`` column (encoder splits).
    ``split_mode=\"random\"`` redraws stratified splits (legacy behavior).
    """
    data = pd.read_csv(csv_path)
    if "sample_id" in data.columns:
        data = data.copy()
        data["sample_id"] = data["sample_id"].astype(str).str.strip()
    feature_cols = [c for c in data.columns if c.startswith("dim_")]

    sample_features: dict[str, np.ndarray] = {}
    for sid, grp in data.groupby("sample_id", sort=False):
        sid_s = str(sid).strip()
        grp = grp.sort_values("timepoint")
        arr = grp[feature_cols].values.astype(np.float64)
        if len(arr) <= time_point_idx:
            continue
        sample_features[sid_s] = arr[time_point_idx]

    def pack(id_list: list[str]) -> tuple[pd.DataFrame, pd.Series]:
        X = pd.DataFrame(
            [sample_features[i] for i in id_list],
            index=id_list,
            columns=feature_cols,
        )
        y = pd.Series([label_fn(i) for i in id_list], index=id_list, dtype=int)
        return X, y

    if split_mode == "csv":
        train_ids, val_ids, test_ids = train_val_test_ids_from_encoded_csv(data)
        missing = [i for i in train_ids + val_ids + test_ids if i not in sample_features]
        if missing:
            raise ValueError(
                f"{csv_path}: split lists sample_ids with no row at time_point_idx={time_point_idx}: "
                f"{missing[:8]!r}{'...' if len(missing) > 8 else ''}"
            )
    else:
        ids = list(sample_features.keys())
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

    X_tr, y_tr = pack(train_ids)
    X_va, y_va = pack(val_ids)
    X_te, y_te = pack(test_ids)
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def _emergent_join_key(sample_id: Any, timepoint: Any) -> tuple[str, float]:
    return (str(sample_id).strip(), round(float(timepoint), 6))


def resolve_emergent_target_column(columns: Sequence[str], property_name: str) -> str:
    """
    Map a short name like ``\"GROWTH\"`` to ``property_GROWTH`` when present on the emergents CSV.
    """
    name = property_name.strip()
    pu = name.upper()
    cand = f"property_{pu}"
    if cand in columns:
        return cand
    if name in columns:
        return name
    if pu in columns:
        return pu
    raise ValueError(
        f"No emergent column for {property_name!r}; expected {cand!r} or a matching column name "
        f"(available: {sorted(columns)})"
    )


def emergent_target_lookup_from_csv(
    emergents_csv: Path, target_column: str
) -> dict[tuple[str, float], float]:
    """
    Finite targets only: NaN, empty string, ``nan`` / ``none`` (case-insensitive) are omitted, so those
    (sample_id, timepoint) pairs are skipped during training.
    """
    df = pd.read_csv(emergents_csv)
    df = df.copy()
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    for col in ("sample_id", "timepoint", target_column):
        if col not in df.columns:
            raise ValueError(f"{emergents_csv}: missing column {col!r}")
    out: dict[tuple[str, float], float] = {}
    dup = df.duplicated(subset=["sample_id", "timepoint"], keep=False)
    if dup.any():
        dfn = df[["sample_id", "timepoint"]].loc[dup].drop_duplicates()
        raise ValueError(
            f"{emergents_csv}: duplicate sample_id+timepoint rows ({dup.sum()} rows); "
            f"resolve before regression (example keys: {dfn.head(3).to_dict('records')})"
        )
    for _, row in df.iterrows():
        sid = row["sample_id"]
        tp_raw = row["timepoint"]
        y_raw = row[target_column]
        if pd.isna(tp_raw):
            continue
        try:
            tp = float(tp_raw)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(tp):
            continue
        if pd.isna(y_raw):
            continue
        if isinstance(y_raw, str):
            s = y_raw.strip().lower()
            if s in ("", "nan", "none", "null"):
                continue
        try:
            yv = float(y_raw)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(yv):
            continue
        out[_emergent_join_key(sid, tp)] = yv
    return out


def _emergent_example_key(sample_id: str, timepoint: float) -> tuple[str, float]:
    return (str(sample_id).strip(), round(float(timepoint), 6))


def load_latent_train_val_test_emergent_regression(
    csv_path: Path,
    time_point_idx: int,
    emergent_lookup: dict[tuple[str, float], float],
    random_seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.2,
    split_mode: str = "csv",
    target_timepoints: Sequence[float] | None = None,
    timepoint_atol: float = 1e-5,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    dict[str, Any],
]:
    """
    Latent features joined to ``emergent_lookup`` on ``(sample_id, timepoint)``.

    **Explicit simulation times:** pass ``target_timepoints=[1.0, …, 15.0]`` for one training row per
    (sample, day) where a finite emergent exists (split is still by ``sample_id``).

    **Index mode:** ``target_timepoints`` is ``None`` or empty → ``iloc[time_point_idx]`` (one row per
    sample).
    """
    if target_timepoints is not None and len(target_timepoints) > 0:
        ttps: list[float] | None = [float(x) for x in target_timepoints]
    else:
        ttps = None

    data = pd.read_csv(csv_path)
    if "sample_id" in data.columns:
        data = data.copy()
        data["sample_id"] = data["sample_id"].astype(str).str.strip()
    feature_cols = [c for c in data.columns if c.startswith("dim_")]
    if "timepoint" not in data.columns:
        raise ValueError(f"{csv_path}: missing 'timepoint' column")
    if not feature_cols:
        raise ValueError(
            f"{csv_path}: no dim_* latent columns; cannot build features for emergent regression."
        )

    enc_sids = set(data["sample_id"].unique())
    em_sids = {k[0] for k in emergent_lookup}
    if not enc_sids & em_sids:
        raise ValueError(
            f"{csv_path}: no sample_id overlap between this encoded_data.csv and the emergents table "
            f"(encoded has {len(enc_sids)} ids, emergents {len(em_sids)}). "
            f"Example encoded ids: {sorted(enc_sids)[:5]!r}; "
            f"example emergent ids: {sorted(em_sids)[:5]!r}. "
            "Use an emergents CSV built for the same dataset as the encoder run "
            "(e.g. ``arcade_emergents.csv`` only aligns with ARCADE ``sample_id`` values like "
            "``CH_Lav_0``, not gastruloid ``array*`` ids)."
        )

    n_no_latent_row = 0
    n_missing_target = 0
    sample_features: dict[tuple[str, float], np.ndarray] = {}
    sample_targets: dict[tuple[str, float], float] = {}

    for sid, grp in data.groupby("sample_id", sort=False):
        sid_s = str(sid).strip()
        grp = grp.sort_values("timepoint")
        t_col = pd.to_numeric(grp["timepoint"], errors="coerce")

        if ttps is not None:
            for ttp in ttps:
                diff = (t_col - float(ttp)).abs()
                ok = t_col.notna() & (diff <= timepoint_atol)
                if not ok.any():
                    n_no_latent_row += 1
                    continue
                first_i = int(np.flatnonzero(ok.to_numpy())[0])
                row = grp.iloc[first_i]
                try:
                    tp = float(row["timepoint"])
                except (TypeError, ValueError):
                    n_no_latent_row += 1
                    continue
                lk = _emergent_join_key(sid_s, tp)
                if lk not in emergent_lookup:
                    n_missing_target += 1
                    continue
                pk = _emergent_example_key(sid_s, tp)
                sample_features[pk] = row[feature_cols].values.astype(np.float64)
                sample_targets[pk] = emergent_lookup[lk]
        else:
            if len(grp) <= time_point_idx:
                n_no_latent_row += 1
                continue
            row = grp.iloc[time_point_idx]
            try:
                tp = float(row["timepoint"])
            except (TypeError, ValueError):
                n_no_latent_row += 1
                continue
            lk = _emergent_join_key(sid_s, tp)
            if lk not in emergent_lookup:
                n_missing_target += 1
                continue
            pk = _emergent_example_key(sid_s, tp)
            sample_features[pk] = row[feature_cols].values.astype(np.float64)
            sample_targets[pk] = emergent_lookup[lk]

    def pack(id_keys: list[tuple[str, float]]) -> tuple[pd.DataFrame, pd.Series]:
        mi = pd.MultiIndex.from_tuples(id_keys, names=["sample_id", "timepoint"])
        X = pd.DataFrame(
            [sample_features[k] for k in id_keys],
            index=mi,
            columns=feature_cols,
        )
        y = pd.Series([sample_targets[k] for k in id_keys], index=mi, dtype=float)
        return X, y

    loc = csv_path

    if split_mode == "csv":
        train_sids, val_sids, test_sids = train_val_test_ids_from_encoded_csv(data)
        train_sid_set = set(train_sids)
        val_sid_set = set(val_sids)
        test_sid_set = set(test_sids)
        seen_sids = {k[0] for k in sample_features}
        train_sids_eff = sorted(train_sid_set & seen_sids)
        val_sids_eff = sorted(val_sid_set & seen_sids)
        test_sids_eff = sorted(test_sid_set & seen_sids)
        empty = [
            (n, s)
            for n, s in [
                ("train", train_sids_eff),
                ("val", val_sids_eff),
                ("test", test_sids_eff),
            ]
            if not s
        ]
        if empty:
            hint = ""
            if len(sample_features) == 0:
                n_enc = int(data["sample_id"].nunique())
                ex_em = next(iter(emergent_lookup.keys()), None)
                hint = (
                    f" No (sample_id, timepoint) row joined encodings to emergents "
                    f"({n_enc} encoded sample_ids, example emergent key {ex_em!r}). "
                    "Check sample_id spelling matches arcade_emergents.csv and timepoints exist in both."
                )
            elif not seen_sids:
                hint = " seen_sids empty (unexpected)."
            else:
                hint = (
                    f" Example sample_id in encodings with a target: {next(iter(seen_sids))!r}; "
                    f"in train split list: {next(iter(train_sid_set), None)!r}."
                )
            if n_missing_target > 0 and ttps is None and time_point_idx == 0:
                hint += (
                    " Encoded CSV may start at day 0 / half-day steps while emergents use days 1–15; "
                    "set target_timepoints=[1.0,…,15.0]."
                )
            raise ValueError(
                f"{loc}: after requiring finite emergent targets, empty splits {empty!r} "
                f"(n_emergent_rows={len(sample_features)}, "
                f"no_matching_encoded_or_time_row={n_no_latent_row}, "
                f"missing_or_nan_emergent={n_missing_target}).{hint}"
            )
        train_keys = sorted(k for k in sample_features if k[0] in train_sid_set)
        val_keys = sorted(k for k in sample_features if k[0] in val_sid_set)
        test_keys = sorted(k for k in sample_features if k[0] in test_sid_set)
    else:
        sample_level = sorted({k[0] for k in sample_features})
        train_sids, test_sids = train_test_split(
            sample_level,
            test_size=test_size,
            random_state=random_seed,
        )
        train_sids, val_sids = train_test_split(
            train_sids,
            test_size=val_size,
            random_state=random_seed,
        )
        ts_tr, ts_va, ts_te = set(train_sids), set(val_sids), set(test_sids)
        train_keys = sorted(k for k in sample_features if k[0] in ts_tr)
        val_keys = sorted(k for k in sample_features if k[0] in ts_va)
        test_keys = sorted(k for k in sample_features if k[0] in ts_te)

    meta: dict[str, Any] = {
        "n_excluded_no_encoded_time_row": n_no_latent_row,
        "n_excluded_missing_or_nan_emergent": n_missing_target,
        "n_with_emergent_target_rows": len(sample_features),
        "target_timepoints_effective": ttps,
    }
    X_tr, y_tr = pack(train_keys)
    X_va, y_va = pack(val_keys)
    X_te, y_te = pack(test_keys)
    return X_tr, y_tr, X_va, y_va, X_te, y_te, meta


def run_regressor_on_splits(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    regressor_type: str = "linear_regression",
    cv_folds: int = 5,
    eval_metric: str = "r2",
) -> dict[str, Any]:
    """Grid search on train; refit on train+val; return test metrics and per-test predictions."""
    reg = SupervisedRegressor(model_type=regressor_type, eval_metric=eval_metric)
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_val_s = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    reg.grid_search(X_train_s, y_train, cv_folds=cv_folds)
    X_fit = pd.concat([X_train_s, X_val_s])
    y_fit = pd.concat([y_train, y_val])
    reg.fit(X_fit, y_fit)

    y_pred = reg.predict(X_test_s)
    idx = X_test.index
    yt = y_test.loc[idx].values.astype(float)
    out: dict[str, Any] = {
        "test_mse": float(mean_squared_error(yt, y_pred)),
        "test_mae": float(mean_absolute_error(yt, y_pred)),
        "test_r2": float(r2_score(yt, y_pred)),
        "test_y_true": [float(x) for x in yt],
        "test_y_pred": [float(x) for x in y_pred],
    }
    if isinstance(idx, pd.MultiIndex) and idx.nlevels == 2:
        out["test_sample_ids"] = [str(t[0]) for t in idx]
        out["test_timepoints"] = [float(t[1]) for t in idx]
    else:
        out["test_sample_ids"] = [str(i) for i in idx]
        out["test_timepoints"] = None
    return out


def run_study_point_emergent_regression(
    study_dir: Path,
    emergents_csv: str | Path,
    emergent_property_name: str,
    study_name: str | None = None,
    time_point_idx: int = 0,
    target_timepoints: Sequence[float] | None = None,
    timepoint_atol: float = 1e-5,
    regressor_types: Sequence[str] = ("linear_regression",),
    random_seed: int = 42,
    cv_folds: int = 5,
    split_mode: str = "csv",
    eval_metric: str = "r2",
) -> dict[str, Any]:
    """
    Point regression: predict an emergent from latents at one or many simulation times.

    Pass ``target_timepoints=(1.0,…,15.0)`` for one training row per sample per day (when the
    emergent is finite). Splits follow ``sample_id`` so all days of a trajectory stay in the same fold.
    If ``target_timepoints`` is omitted or empty, uses ``time_point_idx`` only (one row per sample).

    Rows with non-finite targets in the emergents table are omitted from ``emergent_lookup``; encoded
    rows whose ``(sample_id, timepoint)`` has no finite target are skipped for that day.
    """
    study_dir = study_dir.expanduser().resolve()
    emergents_csv = Path(emergents_csv).expanduser().resolve()
    name = study_name or study_dir.name
    header = pd.read_csv(emergents_csv, nrows=0)
    target_col = resolve_emergent_target_column(list(header.columns), emergent_property_name)
    emergent_lookup = emergent_target_lookup_from_csv(emergents_csv, target_col)

    reg_list = list(regressor_types)
    if not reg_list:
        raise ValueError("regressor_types must be a non-empty sequence")

    rows: list[dict[str, Any]] = []
    for model_name, dataset_name, csv_path in iter_encoded_csv_paths(study_dir):
        Xt = load_latent_train_val_test_emergent_regression(
            csv_path,
            time_point_idx,
            emergent_lookup,
            random_seed=random_seed,
            split_mode=split_mode,
            target_timepoints=target_timepoints,
            timepoint_atol=timepoint_atol,
        )
        X_tr, y_tr, X_va, y_va, X_te, y_te, ex_meta = Xt
        for reg_t in reg_list:
            print(f"Running {reg_t} on {model_name} {dataset_name}")
            metrics = run_regressor_on_splits(
                X_tr,
                y_tr,
                X_va,
                y_va,
                X_te,
                y_te,
                regressor_type=reg_t,
                cv_folds=cv_folds,
                eval_metric=eval_metric,
            )
            rows.append(
                {
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "regressor_type": reg_t,
                    **ex_meta,
                    **metrics,
                }
            )

    return {
        "study_name": name,
        "model_dataset_results": rows,
        "meta": {
            "results_study_dir": str(study_dir),
            "emergents_csv": str(emergents_csv),
            "emergent_property_name": emergent_property_name,
            "emergent_target_column": target_col,
            "time_point_idx": time_point_idx,
            "target_timepoints": list(target_timepoints) if target_timepoints is not None else None,
            "timepoint_atol": timepoint_atol,
            "regressor_types": reg_list,
            "random_seed": random_seed,
            "split_mode": split_mode,
            "eval_metric": eval_metric,
        },
    }


def run_classifier_on_splits(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
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
    X_val_s = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
    X_test_s = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    clf.grid_search(X_train_s, y_train, cv_folds=cv_folds)
    X_fit = pd.concat([X_train_s, X_val_s])
    y_fit = pd.concat([y_train, y_val])
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
    study_name: str | None = None,
    label_fn: Callable[[str], int] | None = None,
    time_point_idx: int = 0,
    classifier_types: Sequence[str] = ("logistic_regression",),
    random_seed: int = 42,
    cv_folds: int = 5,
    split_mode: str = "csv",
) -> dict[str, Any]:
    """
    For each ``<study_dir>/<encoder>/_best_model/<dataset>/encoded_data.csv``, train classifier(s)
    and record test-set correct/incorrect sample ids.

    ``classifier_types`` must be non-empty; each entry runs a small in-model ``grid_search``.
    """
    study_dir = study_dir.expanduser().resolve()
    name = study_name or study_dir.name
    label_fn = label_fn or gastruloid_binary_label

    clf_list = list(classifier_types)
    if not clf_list:
        raise ValueError("classifier_types must be a non-empty sequence")

    rows: list[dict[str, Any]] = []
    for model_name, dataset_name, csv_path in iter_encoded_csv_paths(study_dir):
        X_tr, y_tr, X_va, y_va, X_te, y_te = load_latent_train_val_test(
            csv_path,
            time_point_idx,
            label_fn,
            random_seed=random_seed,
            split_mode=split_mode,
        )
        for clf_t in clf_list:
            correct, incorrect, _ = run_classifier_on_splits(
                X_tr,
                y_tr,
                X_va,
                y_va,
                X_te,
                y_te,
                classifier_type=clf_t,
                cv_folds=cv_folds,
            )
            rows.append(
                {
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "classifier_type": clf_t,
                    "total_correct": len(correct),
                    "total_incorrect": len(incorrect),
                    "correct_predictions": sorted(correct),
                    "incorrect_predictions": sorted(incorrect),
                }
            )

    return {
        "study_name": name,
        "model_dataset_results": rows,
        "meta": {
            "results_study_dir": str(study_dir),
            "time_point_idx": time_point_idx,
            "classifier_types": clf_list,
            "random_seed": random_seed,
            "label": _label_meta_name(label_fn),
            "split_mode": split_mode,
        },
    }


def run_study_trajectory_classification(
    study_dir: Path,
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
    split_mode: str = "csv",
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
            split_mode=split_mode,
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

        correct, incorrect = _trajectory_classifier_test_split(
            best_model, test_dl, label_fn, device
        )
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
            "label": _label_meta_name(label_fn),
        },
    }


def build_trajectory_loaders(
    results_path: str | Path,
    seq_len: int | None = None,
    *,
    val_split: float = 0.2,
    test_split: float = 0.2,
    random_seed: int = 42,
    split_mode: str = "csv",
) -> dict[str, dict[str, SequenceLoader]]:
    """One :class:`~latent_model.loaders.sequence_loader.SequenceLoader` per ``_best_model`` dataset.

    Default ``split_mode=\"csv\"`` follows the encoder CSV ``split`` column; use ``\"random\"`` to redraw.
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
                split_mode=split_mode,
            )
    return loaders


def temporal_model_sweep(
    input_size: int,
    output_size: int,
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


def _default_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
