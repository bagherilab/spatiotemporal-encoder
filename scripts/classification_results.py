"""Downstream classification and point regression on encoder latents under ``results/<study_name>/``.

- `BACKEND = "point"`: sklearn head on latent vectors at one time point
- `BACKEND = "trajectory"`: RNN/LSTM on the full latent sequence
"""

import json
import math
import statistics
import sys
import tempfile
from pathlib import Path
from collections.abc import Callable, Hashable
from typing import Any

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from latent_model.runner import (
    arcade_colony_tissue_label,
    gastruloid_binary_label,
    run_study_classification,
    run_study_point_emergent_regression,
    run_study_trajectory_classification,
)

def main() -> None:
    repo = _REPO

    RESULTS_STUDY_DIR = repo / "results" / "architecture-gastruloid"
    EMERGENT_TARGET_TIMEPOINTS = [float(d) for d in range(1, 15)]

    # "point" | "trajectory"
    BACKEND = "point"

    # --- point configs ---
    # "classification" | "regression"
    POINT_TASK = "regression"
    CLASSIFIER_TYPES = [
        "logistic_regression",
        "random_forest",
        "svm",
    ]
    TIME_POINT_IDX = 0
    CV_FOLDS = 5

    

    # Point regression (ARCADE): emergents CSV
    EMERGENTS_CSV = repo / "data" / "vascular_function_128" / "labels" / "arcade_emergents.csv"
    # One target or a list; all are run sequentially (one JSON per target).
    EMERGENT_PROPERTY_NAME: str | list[str] = ["VOLUMES", "ACTIVITY", "DIAMETERS", "GROWTH", "SYMMETRY", "COUNTS"]  # e.g. ["VOLUMES", "ACTIVITY"]
    # Point regression (gastruloid): morphometric properties from segmentation.csv
    GASTRULOID_SEGMENTATION_CSV = repo / "data" / "gastruloid_128" / "labels" / "segmentations.csv"
    GASTRULOID_PROPERTY_NAME: str | list[str] = ["Mean", "Area", "XM", "YM", "Circ."]  # e.g. ["Mean", "Area", "AR"]
    REGRESSOR_TYPES = [
        "linear_regression",
        "elastic_net",
        "random_forest",
        # "svr",
        "mlp"
    ]
    REGRESSION_EVAL_METRIC = "r2"

    # --- trajectory configs ---
    TRAJECTORY_CELL = "lstm"  # "lstm" | "rnn"
    TRAJECTORY_NUM_CLASSES = 2
    TRAJECTORY_MAX_EPOCHS = 50
    TRAJECTORY_PATIENCE = 5

    RANDOM_SEED = 42
    NUM_REPEAT_RUNS = 1

    summary_kind: str | None = None
    regression_metric_for_summary: str | None = None

    try:
        study_dir = RESULTS_STUDY_DIR.expanduser().resolve()
        label_fn = label_fn_for_study_dir(study_dir)
        raw_runs: list[dict[str, Any]] = []
        seeds = [RANDOM_SEED + k for k in range(NUM_REPEAT_RUNS)]

        if BACKEND == "point":
            if POINT_TASK == "classification":
                summary_kind = "classification"
                print(f"Classification label: {label_fn.__name__} (study dir {study_dir.name!r})")
                for seed in seeds:
                    print(f"  run seed={seed}")
                    raw_runs.append(
                        run_study_classification(
                            study_dir,
                            label_fn=label_fn,
                            time_point_idx=TIME_POINT_IDX,
                            classifier_types=CLASSIFIER_TYPES,
                            random_seed=seed,
                            cv_folds=CV_FOLDS,
                        )
                    )
                stem = f"{raw_runs[0]['study_name']}_classification_results"
            elif POINT_TASK == "regression":
                summary_kind = "regression"
                regression_metric_for_summary = REGRESSION_EVAL_METRIC
                is_gastruloid = "gastruloid" in study_dir.name.lower()
                if is_gastruloid:
                    seg_csv = GASTRULOID_SEGMENTATION_CSV.expanduser().resolve()
                    prop_specs = _coerce_property_names(GASTRULOID_PROPERTY_NAME)
                    src_label = "gastruloid segmentations"
                else:
                    arcade_csv = EMERGENTS_CSV.expanduser().resolve()
                    tps_arcade = EMERGENT_TARGET_TIMEPOINTS
                    prop_specs = _coerce_property_names(EMERGENT_PROPERTY_NAME)
                    src_label = "arcade emergents"

                out_dir = repo / "results_downstream" / "classification_results"
                out_dir.mkdir(parents=True, exist_ok=True)

                for prop_spec in prop_specs:
                    tmp_property_csv: Path | None = None
                    try:
                        if is_gastruloid:
                            tmp_property_csv, tps, resolved_prop = _build_gastruloid_property_csv(
                                seg_csv,
                                prop_spec,
                            )
                            esc = tmp_property_csv
                            prop_name = resolved_prop
                        else:
                            esc = arcade_csv
                            tps = tps_arcade
                            prop_name = prop_spec

                        if tps:
                            tps_msg = f"{len(tps)} timepoints ({tps[0]!r} … {tps[-1]!r})"
                        else:
                            tps_msg = "time_point_idx / iloc (no explicit days)"
                        print(
                            f"Point regression: property={prop_name!r} from {src_label} {esc} "
                            f"({tps_msg}, study dir {study_dir.name!r})"
                        )
                        raw_runs = []
                        for seed in seeds:
                            print(f"  run seed={seed}")
                            raw_runs.append(
                                run_study_point_emergent_regression(
                                    study_dir,
                                    emergents_csv=esc,
                                    emergent_property_name=prop_name,
                                    time_point_idx=TIME_POINT_IDX,
                                    target_timepoints=tps,
                                    regressor_types=REGRESSOR_TYPES,
                                    random_seed=seed,
                                    cv_folds=CV_FOLDS,
                                    eval_metric=REGRESSION_EVAL_METRIC,
                                )
                            )
                        raw = _merge_repeat_exports(raw_runs, summary_kind, seeds)
                        safe_prop = str(prop_name).replace("/", "_")
                        stem = f"{raw_runs[0]['study_name']}_point_regression_{safe_prop.lower()}"
                        cls_path = out_dir / f"{stem}.json"
                        with open(cls_path, "w", encoding="utf-8") as f:
                            json.dump(raw, f)
                        n = len(raw["model_dataset_results"])
                        print(f"Wrote {cls_path} ({n} model/dataset rows, backend={BACKEND})")
                        _print_best_downstream_result(
                            raw,
                            summary_kind=summary_kind,
                            regression_eval_metric=regression_metric_for_summary,
                        )
                    finally:
                        if tmp_property_csv is not None and tmp_property_csv.is_file():
                            tmp_property_csv.unlink(missing_ok=True)

                return
            else:
                raise ValueError(f"Unknown POINT_TASK: {POINT_TASK!r}")
        elif BACKEND == "trajectory":
            summary_kind = "trajectory"
            print(f"Classification label: {label_fn.__name__} (study dir {study_dir.name!r})")
            for seed in seeds:
                print(f"  run seed={seed}")
                raw_runs.append(
                    run_study_trajectory_classification(
                        study_dir,
                        label_fn=label_fn,
                        random_seed=seed,
                        cell=TRAJECTORY_CELL,
                        num_classes=TRAJECTORY_NUM_CLASSES,
                        max_epochs=TRAJECTORY_MAX_EPOCHS,
                        patience=TRAJECTORY_PATIENCE,
                    )
                )
            stem = f"{raw_runs[0]['study_name']}_trajectory_classification_results"
        else:
            raise ValueError(f"Unknown BACKEND: {BACKEND!r}")

        raw = _merge_repeat_exports(raw_runs, summary_kind, seeds)

        out_dir = repo / "results_downstream" / "classification_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        cls_path = out_dir / f"{stem}.json"
        with open(cls_path, "w", encoding="utf-8") as f:
            json.dump(raw, f)

        n = len(raw["model_dataset_results"])
        print(f"Wrote {cls_path} ({n} model/dataset rows, backend={BACKEND})")

        if summary_kind is not None:
            _print_best_downstream_result(
                raw,
                summary_kind=summary_kind,
                regression_eval_metric=regression_metric_for_summary,
            )

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _coerce_property_names(names: str | list[str]) -> list[str]:
    """Single string or non-empty list of target column names for emergent / segmentation regression."""
    if isinstance(names, str):
        s = names.strip()
        if not s:
            raise ValueError("property name must be non-empty")
        return [s]
    if not names:
        raise ValueError("property names list must be non-empty")
    out = [str(x).strip() for x in names]
    if not all(out):
        raise ValueError("property names must be non-empty strings")
    return out


def _test_accuracy(row: dict) -> float:
    if "test_accuracy_mean" in row:
        v = row["test_accuracy_mean"]
        return float(v) if v is not None else float("nan")
    accs = row.get("test_accuracies")
    if isinstance(accs, list) and accs:
        return float(statistics.fmean(float(x) for x in accs))
    c, w = row.get("total_correct", 0), row.get("total_incorrect", 0)
    n = c + w
    return float(c) / n if n else float("nan")


def _accuracy_from_counts(row: dict) -> float:
    c, w = row.get("total_correct", 0), row.get("total_incorrect", 0)
    n = c + w
    return float(c) / n if n else float("nan")


def _strip_sample_level_classification(row: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in row.items() if k not in ("correct_predictions", "incorrect_predictions")}
    return out


def _merge_repeat_exports(
    raw_runs: list[dict[str, Any]],
    summary_kind: str | None,
    seeds: list[int],
) -> dict[str, Any]:
    """
    Combine repeated runs into one JSON-friendly dict: per-row repeat metrics (for error bars),
    no per-sample correct/incorrect or y_true/y_pred lists.
    """
    if not raw_runs:
        raise ValueError("raw_runs is empty")
    base = raw_runs[0]
    out: dict[str, Any] = {
        "study_name": base.get("study_name"),
        "meta": {**(base.get("meta") or {}), "repeat_seeds": seeds, "num_repeat_runs": len(seeds)},
        "model_dataset_results": [],
    }

    if summary_kind == "classification":
        key_fn: Callable[[dict[str, Any]], tuple[Hashable, ...]] = lambda r: (
            r.get("model_name"),
            r.get("dataset_name"),
            r.get("classifier_type"),
        )
        out["model_dataset_results"] = _merge_repeat_classification_rows(raw_runs, key_fn)
        return out

    if summary_kind == "trajectory":
        key_fn = lambda r: (r.get("model_name"), r.get("dataset_name"))
        out["meta"]["sample_id_predictions_repeat_index"] = 0
        out["meta"]["sample_id_predictions_seed"] = seeds[0] if seeds else None
        out["model_dataset_results"] = _merge_repeat_classification_rows(
            raw_runs,
            key_fn,
            attach_first_run_prediction_ids=True,
        )
        return out

    if summary_kind == "regression":
        key_fn = lambda r: (r.get("model_name"), r.get("dataset_name"), r.get("regressor_type"))
        eval_metric = str((base.get("meta") or {}).get("eval_metric", "r2"))
        out["meta"]["parity_predictions_repeat_index"] = 0
        out["meta"]["parity_predictions_seed"] = seeds[0] if seeds else None
        out["meta"]["parity_predictions_eval_metric"] = eval_metric
        out["model_dataset_results"] = _merge_repeat_regression_rows(
            raw_runs,
            key_fn,
            eval_metric=eval_metric,
        )
        return out

    out["model_dataset_results"] = base.get("model_dataset_results") or []
    return out


def _merge_repeat_classification_rows(
    raw_runs: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], tuple[Hashable, ...]],
    attach_first_run_prediction_ids: bool = False,
) -> list[dict[str, Any]]:
    all_keys: set[tuple[Hashable, ...]] = set()
    per_run: list[dict[tuple[Hashable, ...], dict[str, Any]]] = []
    for run in raw_runs:
        idx = {key_fn(r): r for r in (run.get("model_dataset_results") or [])}
        per_run.append(idx)
        all_keys.update(idx.keys())

    merged: list[dict[str, Any]] = []
    for k in sorted(all_keys, key=lambda t: tuple(str(x) for x in t)):
        accs: list[float] = []
        base_row: dict[str, Any] | None = None
        for idx in per_run:
            row = idx.get(k)
            if row is None:
                continue
            accs.append(_accuracy_from_counts(row))
            base_row = _strip_sample_level_classification(row)
        if base_row is None:
            continue
        m = dict(base_row)
        m["test_accuracies"] = accs
        m["test_accuracy_mean"] = float(statistics.fmean(accs)) if accs else float("nan")
        m["test_accuracy_std"] = float(statistics.stdev(accs)) if len(accs) > 1 else 0.0
        m.pop("total_correct", None)
        m.pop("total_incorrect", None)
        if attach_first_run_prediction_ids and per_run:
            first_row = per_run[0].get(k)
            if first_row is not None:
                cp = first_row.get("correct_predictions")
                ip = first_row.get("incorrect_predictions")
                if cp is not None:
                    m["correct_predictions"] = list(cp)
                if ip is not None:
                    m["incorrect_predictions"] = list(ip)
        merged.append(m)
    return merged


def _merge_repeat_regression_rows(
    raw_runs: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], tuple[Hashable, ...]],
    *,
    eval_metric: str = "r2",
) -> list[dict[str, Any]]:
    drop_keys = frozenset(
        {"test_y_true", "test_y_pred", "test_sample_ids", "test_timepoints"}
    )
    all_keys: set[tuple[Hashable, ...]] = set()
    per_run: list[dict[tuple[Hashable, ...], dict[str, Any]]] = []
    for run in raw_runs:
        idx = {key_fn(r): r for r in (run.get("model_dataset_results") or [])}
        per_run.append(idx)
        all_keys.update(idx.keys())

    merged_by_key: dict[tuple[Hashable, ...], dict[str, Any]] = {}
    for k in sorted(all_keys, key=lambda t: tuple(str(x) for x in t)):
        r2s: list[float] = []
        maes: list[float] = []
        mses: list[float] = []
        base_row: dict[str, Any] | None = None
        for idx in per_run:
            row = idx.get(k)
            if row is None:
                continue
            r2s.append(float(row["test_r2"]))
            maes.append(float(row["test_mae"]))
            mses.append(float(row["test_mse"]))
            base_row = {x: v for x, v in row.items() if x not in drop_keys}
        if base_row is None:
            continue
        m = dict(base_row)
        m["test_r2_values"] = r2s
        m["test_r2_mean"] = float(statistics.fmean(r2s)) if r2s else float("nan")
        m["test_r2_std"] = float(statistics.stdev(r2s)) if len(r2s) > 1 else 0.0
        m["test_mae_values"] = maes
        m["test_mae_mean"] = float(statistics.fmean(maes)) if maes else float("nan")
        m["test_mae_std"] = float(statistics.stdev(maes)) if len(maes) > 1 else 0.0
        m["test_mse_values"] = mses
        m["test_mse_mean"] = float(statistics.fmean(mses)) if mses else float("nan")
        m["test_mse_std"] = float(statistics.stdev(mses)) if len(mses) > 1 else 0.0
        # Single scalar fields for downstream code / tables
        m["test_r2"] = m["test_r2_mean"]
        m["test_mae"] = m["test_mae_mean"]
        m["test_mse"] = m["test_mse_mean"]
        merged_by_key[k] = m

    if not merged_by_key:
        return []

    metric = str(eval_metric).strip().lower()

    def _metric_value(row: dict[str, Any]) -> float:
        if metric == "r2":
            v = row.get("test_r2")
            vf = float(v) if v is not None else float("nan")
            return vf if math.isfinite(vf) else -float("inf")
        if metric == "mae":
            v = row.get("test_mae")
            vf = float(v) if v is not None else float("nan")
            return vf if math.isfinite(vf) else float("inf")
        if metric in ("mse", "rmse"):
            v = row.get("test_mse")
            vf = float(v) if v is not None else float("nan")
            return vf if math.isfinite(vf) else float("inf")
        v = row.get("test_r2")
        vf = float(v) if v is not None else float("nan")
        return vf if math.isfinite(vf) else -float("inf")

    minimize = metric in ("mae", "mse", "rmse")
    first_run = per_run[0] if per_run else {}
    keys_by_model: dict[str, list[tuple[Hashable, ...]]] = {}
    for key, row in merged_by_key.items():
        mn = str(row.get("model_name"))
        keys_by_model.setdefault(mn, []).append(key)

    for model_name, model_keys in keys_by_model.items():
        if not model_keys:
            continue
        best_key = min(model_keys, key=lambda k: _metric_value(merged_by_key[k])) if minimize else max(
            model_keys, key=lambda k: _metric_value(merged_by_key[k])
        )
        source = first_run.get(best_key)
        if source is None:
            continue
        best_row = merged_by_key[best_key]
        for k in ("test_y_true", "test_y_pred", "test_sample_ids", "test_timepoints"):
            v = source.get(k)
            if v is not None:
                best_row[k] = list(v) if isinstance(v, list) else v
        best_row["has_parity_predictions"] = True

    return [merged_by_key[k] for k in sorted(merged_by_key, key=lambda t: tuple(str(x) for x in t))]

def _print_best_downstream_result(
    raw: dict,
    *,
    summary_kind: str,
    regression_eval_metric: str | None = None,
    ) -> None:
    """Print the single best row in ``model_dataset_results`` for the current task."""
    rows = raw.get("model_dataset_results") or []
    if not rows:
        print("No model/dataset rows in results.")
        return

    model_names = sorted({r.get("model_name") for r in rows if r.get("model_name") is not None})
    if not model_names:
        print("No ``model_name`` values found in results.")
        return

    if summary_kind == "regression":
        m = (regression_eval_metric or (raw.get("meta") or {}).get("eval_metric") or "r2").lower()
        maximize = m == "r2"

        def sort_key(row: dict) -> float:
            if m == "r2":
                v = row.get("test_r2")
                vf = float(v) if v is not None else float("nan")
                return vf if math.isfinite(vf) else -float("inf")
            if m == "mae":
                v = row.get("test_mae")
                vf = float(v) if v is not None else float("nan")
                return vf if math.isfinite(vf) else float("inf")
            if m in ("mse", "rmse"):
                v = row.get("test_mse")
                vf = float(v) if v is not None else float("nan")
                return vf if math.isfinite(vf) else float("inf")
            v = row.get("test_r2")
            vf = float(v) if v is not None else float("nan")
            return vf if math.isfinite(vf) else -float("inf")

        def metric_line(best: dict) -> str:
            if m == "r2":
                return f"test R² = {best.get('test_r2')}"
            if m == "mae":
                return f"test MAE = {best.get('test_mae')}"
            if m == "mse":
                return f"test MSE = {best.get('test_mse')}"
            if m == "rmse":
                te_mse = best.get("test_mse")
                rmse = math.sqrt(float(te_mse)) if te_mse is not None else float("nan")
                return f"test RMSE = {rmse} (from test MSE = {te_mse})"
            return f"test R² = {best.get('test_r2')}"

        print(f"\n--- Best per encoder model (by {m} on held-out test) ---")
        for model_name in model_names:
            model_rows = [r for r in rows if r.get("model_name") == model_name]
            if not model_rows:
                continue
            best = max(model_rows, key=sort_key) if maximize else min(model_rows, key=sort_key)

            head = best.get("regressor_type", "?")
            print(f"\nEncoder: {model_name}")
            print(f"  dataset:     {best.get('dataset_name')}")
            print(f"  regressor:   {head}")
            print(f"  selection:   {metric_line(best)}")
            print(
                "  also:        "
                f"MSE={best.get('test_mse')}, MAE={best.get('test_mae')}, R²={best.get('test_r2')}"
            )
        return

    if summary_kind == "classification":
        print("\n--- Best per encoder model (by test accuracy) ---")
        for model_name in model_names:
            model_rows = [r for r in rows if r.get("model_name") == model_name]
            best = max(model_rows, key=_test_accuracy)
            acc = _test_accuracy(best)
            c, w = best.get("total_correct", 0), best.get("total_incorrect", 0)
            head = best.get("classifier_type", "?")

            print(f"\nEncoder: {model_name}")
            print(f"  dataset:     {best.get('dataset_name')}")
            print(f"  classifier:  {head}")
            std = best.get("test_accuracy_std")
            n_rep = len(best.get("test_accuracies") or [])
            if n_rep > 0 and std is not None:
                print(f"  test acc:    {acc:.4f} ± {std:.4f} (mean over {n_rep} runs)")
            else:
                print(f"  test acc:    {acc:.4f} ({c} correct / {c + w} total)")
        return

    if summary_kind == "trajectory":
        meta = raw.get("meta") or {}
        cell = meta.get("cell", "?")

        print("\n--- Best per encoder model (by test accuracy) ---")
        for model_name in model_names:
            model_rows = [r for r in rows if r.get("model_name") == model_name]
            best = max(model_rows, key=_test_accuracy)
            acc = _test_accuracy(best)
            c, w = best.get("total_correct", 0), best.get("total_incorrect", 0)
            bv = best.get("best_val_loss")
            bv_s = f"{float(bv):.4f}" if bv is not None else "n/a"

            print(f"\nEncoder: {model_name}")
            print(f"  dataset:     {best.get('dataset_name')}")
            print(
                f"  trajectory:  {cell}  "
                f"hidden={best.get('best_hidden_size')}  "
                f"layers={best.get('best_num_layers')}  "
                f"(val_loss={bv_s})"
            )
            std = best.get("test_accuracy_std")
            n_rep = len(best.get("test_accuracies") or [])
            if n_rep > 0 and std is not None:
                print(f"  test acc:    {acc:.4f} ± {std:.4f} (mean over {n_rep} runs)")
            else:
                print(f"  test acc:    {acc:.4f} ({c} correct / {c + w} total)")
        return

    print(f"Unknown summary_kind: {summary_kind!r}")

def label_fn_for_study_dir(study_dir: Path):
    name = study_dir.name.lower()
    if "arcade" in name:
        return arcade_colony_tissue_label
    if "gastruloid" in name:
        return gastruloid_binary_label


def _parse_image_sample_time(image_name: str) -> tuple[str, float]:
    stem = Path(str(image_name)).stem
    sample_id, tp_str = stem.rsplit("_", 1)
    return sample_id.strip(), float(tp_str)


def _normalize_gastruloid_sample_id(sample_id: str) -> str:
    sid = str(sample_id).strip()
    if sid.startswith("green_"):
        sid = sid[len("green_") :]
    parts = sid.split("_")
    if parts and parts[-1].isdigit():
        parts[-1] = str(int(parts[-1]))
    return "_".join(parts)


def _build_gastruloid_property_csv(
    source_csv: Path,
    property_name: str,
    ) -> tuple[Path, list[float], str]:
    """
    Convert gastruloid ``segmentations.csv`` into emergent-like columns:
    ``sample_id,timepoint,<property_column>``.
    """
    df = pd.read_csv(source_csv)
    if "Image" not in df.columns:
        raise ValueError(f"{source_csv}: missing required column 'Image'")

    if property_name in df.columns:
        prop_col = property_name
    elif property_name.upper() in df.columns:
        prop_col = property_name.upper()
    else:
        raise ValueError(
            f"{source_csv}: property {property_name!r} not found. "
            f"Available columns: {list(df.columns)}"
        )

    pairs = df["Image"].map(_parse_image_sample_time)
    parsed = pd.DataFrame(pairs.tolist(), columns=["sample_id", "timepoint"])
    out = pd.concat([parsed, df[[prop_col]].copy()], axis=1)
    out["sample_id"] = out["sample_id"].astype(str).map(_normalize_gastruloid_sample_id)
    out["timepoint"] = pd.to_numeric(out["timepoint"], errors="coerce")
    out[prop_col] = pd.to_numeric(out[prop_col], errors="coerce")
    out = out.dropna(subset=["sample_id", "timepoint", prop_col])
    out = out.drop_duplicates(subset=["sample_id", "timepoint"], keep="last")
    if out.empty:
        raise ValueError(f"{source_csv}: no usable rows after parsing sample/timepoint and property")

    tp_vals = sorted({float(x) for x in out["timepoint"].tolist()})
    tmp = tempfile.NamedTemporaryFile(prefix="gastruloid_props_", suffix=".csv", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    out.to_csv(tmp_path, index=False)
    return tmp_path, tp_vals, prop_col


if __name__ == "__main__":
    main()
