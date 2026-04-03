"""Merge ARCADE vascular JSON labels (``VASCULAR_FUNCTION_*``) into csv file"""

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

# Times in JSON are days; align with full-day images (no day 0, no *.5).
_DEFAULT_IMAGE_DAY_MIN = 1
_DEFAULT_IMAGE_DAY_MAX = 15
_TIME_INT_TOL = 1e-6


def timepoint_matches_image_days(
    t: float,
    *,
    min_day: int = _DEFAULT_IMAGE_DAY_MIN,
    max_day: int = _DEFAULT_IMAGE_DAY_MAX,
    integer_days_only: bool = True,
) -> bool:
    """
    Keep only finite simulation times that match full-day imaging (e.g. days 1–15, not 0 or 1.5).

    When ``integer_days_only`` is True, only values equal to an integer within ``[min_day, max_day]``
    are kept (so 0.0 and half-day steps are dropped).
    """
    try:
        tf = float(t)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(tf):
        return False
    if integer_days_only:
        if abs(tf - round(tf)) > _TIME_INT_TOL:
            return False
        d = int(round(tf))
        return min_day <= d <= max_day
    return min_day <= tf <= max_day


def sample_id_to_group_key_seed(sample_id: str) -> tuple[str, int]:
    """
    ``C_Lav_20`` → (``C_Lav``, 20); ``CH_Lava_1`` → (``CH_Lava``, 1).

    ``sample_id`` is ``{context}_{vasc_type}_{seed}`` as in :class:`ARCADELoader`.
    """
    sid = str(sample_id).strip()
    key, seed_str = sid.rsplit("_", 1)
    return key, int(seed_str)


def _json_scalar_to_float(x: Any) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, str) and x.lower() == "nan":
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _load_label_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_label_time_entries(path: Path) -> list[dict[str, Any]]:
    """
    Return time blocks ``{\"_\": [...], \"time\": t}`` for one JSON file.

    Most labels use a **flat** list of those dicts. ``TYPES`` and ``POPS`` instead use a **top-level
    list of parallel time series** (each element is itself a list of time blocks). Walking every
    branch used to emit the same ``(sample_id, timepoint)`` multiple times (once per parallel
    series), and merging properties then exploded rows (e.g. 7× for types × 2 for pops).

    For multi-series files we keep only the **first** series so there is exactly one value per
    seed and simulation time, consistent with one emergent row per image timepoint.
    """
    raw = _load_label_json(path)
    if not raw or not isinstance(raw, list):
        return []

    def _as_time_blocks(seq: list[Any]) -> list[dict[str, Any]]:
        return [
            x
            for x in seq
            if isinstance(x, dict) and isinstance(x.get("_"), list) and "time" in x
        ]

    first = raw[0]
    if isinstance(first, dict):
        return _as_time_blocks(raw)
    if isinstance(first, list):
        return _as_time_blocks(first)
    return []


def lookup_arcade_property(
    label_dir: Path,
    property_name: str,
    group_key: str,
    time: float,
    seed: int,
    file_prefix: str = "VASCULAR_FUNCTION",
    cache: dict[tuple[str, str], list[dict[str, Any]]] | None = None,
) -> float:
    """
    Read ``{file_prefix}_{group_key}.SEEDS.{PROPERTY}.json`` and return value at ``time`` / ``seed``.
    """
    prop_upper = property_name.upper()
    rel = f"{file_prefix}_{group_key}.SEEDS.{prop_upper}.json"
    path = label_dir / rel
    if not path.is_file():
        return float("nan")

    ck = (prop_upper, group_key)
    if cache is not None:
        if ck not in cache:
            cache[ck] = _load_label_time_entries(path)
        entries = cache[ck]
    else:
        entries = _load_label_time_entries(path)

    for val in entries:
        if val.get("time") == time:
            arr = val.get("_")
            if not isinstance(arr, list) or seed < 0 or seed >= len(arr):
                return float("nan")
            return _json_scalar_to_float(arr[seed])
    return float("nan")


def merge_arcade_property_column(
    df: pd.DataFrame,
    label_dir: str | Path,
    property_name: str,
    column_name: str | None = None,
    file_prefix: str = "VASCULAR_FUNCTION",
    timepoint_col: str = "timepoint",
    sample_id_col: str = "sample_id",
) -> pd.DataFrame:
    label_dir = Path(label_dir)
    if not label_dir.is_dir():
        raise FileNotFoundError(f"label_dir not found: {label_dir}")
    if sample_id_col not in df.columns or timepoint_col not in df.columns:
        raise ValueError(f"DataFrame must include {sample_id_col!r} and {timepoint_col!r}")
    out = df.copy()
    col = column_name or f"property_{property_name.upper()}"
    cache: dict[tuple[str, str], list[dict[str, Any]]] = {}

    values: list[float] = []
    for _, row in out.iterrows():
        try:
            gkey, seed = sample_id_to_group_key_seed(row[sample_id_col])
            t = float(row[timepoint_col])
        except (ValueError, TypeError):
            values.append(float("nan"))
            continue
        values.append(
            lookup_arcade_property(
                label_dir,
                property_name,
                gkey,
                t,
                seed,
                file_prefix=file_prefix,
                cache=cache,
            )
        )
    out[col] = values
    return out


def skeleton_without_latents(encoded_csv: str | Path) -> pd.DataFrame:
    """
    Columns from ``encoded_data.csv`` that are not latent dimensions (everything where name does not
    start with ``dim_``). Always includes ``sample_id`` and ``timepoint``.
    """
    encoded_csv = Path(encoded_csv)
    df = pd.read_csv(encoded_csv)
    meta = [c for c in df.columns if not str(c).startswith("dim_")]
    for required in ("sample_id", "timepoint"):
        if required not in meta:
            raise ValueError(f"{encoded_csv}: missing required column {required!r}")
    return df[meta].copy()


def discover_property_names(
    label_dir: str | Path,
    file_prefix: str = "VASCULAR_FUNCTION",
) -> list[str]:
    """Unique property tokens from ``{file_prefix}_*.SEEDS.{PROPERTY}.json`` filenames."""
    label_dir = Path(label_dir)
    found: set[str] = set()
    for path in label_dir.glob(f"{file_prefix}_*.SEEDS.*.json"):
        stem = path.stem
        if ".SEEDS." not in stem:
            continue
        prop = stem.rsplit(".SEEDS.", 1)[-1]
        if prop:
            found.add(prop.upper())
    return sorted(found)


def add_arcade_properties(
    df: pd.DataFrame,
    label_dir: str | Path,
    property_names: Sequence[str],
    *,
    file_prefix: str = "VASCULAR_FUNCTION",
) -> pd.DataFrame:
    """Append one ``property_*`` column per name (same row order as ``df``)."""
    out = df
    for prop in property_names:
        out = merge_arcade_property_column(out, label_dir, prop, file_prefix=file_prefix)
    return out


def labels_long_from_label_dir(
    label_dir: Path,
    property_names: Sequence[str],
    *,
    file_prefix: str = "VASCULAR_FUNCTION",
    image_timepoints_only: bool = True,
    image_day_min: int = _DEFAULT_IMAGE_DAY_MIN,
    image_day_max: int = _DEFAULT_IMAGE_DAY_MAX,
    allow_half_day_timepoints: bool = False,
) -> pd.DataFrame:
    """
    Build a long table from JSON files only: ``sample_id``, ``timepoint``, ``property_*``.

    No ``split`` or other encoder metadata unless you join to a reference CSV yourself.
    """
    label_dir = Path(label_dir)
    if not label_dir.is_dir():
        raise FileNotFoundError(f"label_dir not found: {label_dir}")

    combined: pd.DataFrame | None = None
    for prop in property_names:
        pu = prop.upper()
        pattern = f"{file_prefix}_*.SEEDS.{pu}.json"
        rows: list[dict[str, Any]] = []
        for path in sorted(label_dir.glob(pattern)):
            stem = path.stem
            suffix = f".SEEDS.{pu}"
            pref = f"{file_prefix}_"
            if not stem.startswith(pref) or not stem.endswith(suffix):
                continue
            gkey = stem[len(pref) : -len(suffix)]
            for val in _load_label_time_entries(path):
                arr = val.get("_")
                t_raw = val.get("time")
                if not isinstance(arr, list) or t_raw is None:
                    continue
                try:
                    tf = float(t_raw)
                except (TypeError, ValueError):
                    continue
                if image_timepoints_only and not timepoint_matches_image_days(
                    tf,
                    min_day=image_day_min,
                    max_day=image_day_max,
                    integer_days_only=not allow_half_day_timepoints,
                ):
                    continue
                for seed, x in enumerate(arr):
                    rows.append(
                        {
                            "group_key": gkey,
                            "seed": seed,
                            "sample_id": f"{gkey}_{seed}",
                            "timepoint": tf,
                            f"property_{pu}": _json_scalar_to_float(x),
                        }
                    )
        df_prop = pd.DataFrame(rows)
        if df_prop.empty:
            continue
        pcol = f"property_{pu}"
        df_prop = df_prop.drop_duplicates(subset=["sample_id", "timepoint"], keep="first")
        if combined is None:
            combined = df_prop
        else:
            combined = combined.merge(
                df_prop[["sample_id", "timepoint", pcol]],
                on=["sample_id", "timepoint"],
                how="outer",
            )

    if combined is None or combined.empty:
        return pd.DataFrame(columns=["group_key", "seed", "sample_id", "timepoint"])
    prop_cols = sorted(c for c in combined.columns if c.startswith("property_"))
    ordered = ["group_key", "seed", "sample_id", "timepoint"] + prop_cols
    ordered = [c for c in ordered if c in combined.columns]
    rest = [c for c in combined.columns if c not in ordered]
    combined = combined[ordered + rest].sort_values(
        ["sample_id", "timepoint"]
    ).reset_index(drop=True)
    return combined


def write_labels_csv(
    output_csv: str | Path,
    label_dir: str | Path,
    property_names: Sequence[str],
    *,
    reference_encoded_csv: str | Path | None = None,
    file_prefix: str = "VASCULAR_FUNCTION",
    image_timepoints_only: bool = True,
    image_day_min: int = _DEFAULT_IMAGE_DAY_MIN,
    image_day_max: int = _DEFAULT_IMAGE_DAY_MAX,
    allow_half_day_timepoints: bool = False,
) -> Path:
    """
    Write a label-only (no ``dim_*``) CSV.

    - If ``reference_encoded_csv`` is set: same rows/columns as that file minus ``dim_*``, plus
      ``property_*`` for each name in ``property_names``.
    - Otherwise: build rows from all matching ``VASCULAR_FUNCTION_*`` JSON files (long format).

    By default, only **integer** simulation days in ``[image_day_min, image_day_max]`` are kept
    (excludes day 0 and half-day steps such as 0.5, 1.5). Use ``--all-timepoints`` for no filter.
    """
    label_dir = Path(label_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if reference_encoded_csv is not None:
        skel = skeleton_without_latents(reference_encoded_csv)
        if image_timepoints_only:
            mask = skel["timepoint"].map(
                lambda x: timepoint_matches_image_days(
                    float(x),
                    min_day=image_day_min,
                    max_day=image_day_max,
                    integer_days_only=not allow_half_day_timepoints,
                )
            )
            skel = skel.loc[mask].copy()
        out = add_arcade_properties(skel, label_dir, property_names, file_prefix=file_prefix)
    else:
        out = labels_long_from_label_dir(
            label_dir,
            property_names,
            file_prefix=file_prefix,
            image_timepoints_only=image_timepoints_only,
            image_day_min=image_day_min,
            image_day_max=image_day_max,
            allow_half_day_timepoints=allow_half_day_timepoints,
        )

    out.to_csv(output_csv, index=False)
    return output_csv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--label-dir",
        type=Path,
        required=True,
        help="Directory containing VASCULAR_FUNCTION_*.SEEDS.*.json",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output CSV path (no latent columns)",
    )
    p.add_argument(
        "--properties",
        type=str,
        default=None,
        help="Comma-separated property names; default: every *.SEEDS.* name under --label-dir",
    )
    p.add_argument(
        "--reference-encoded-csv",
        type=Path,
        default=None,
        help="Optional encoded_data.csv: keep its non-dim columns and rows, add properties",
    )
    p.add_argument(
        "--file-prefix",
        type=str,
        default="VASCULAR_FUNCTION",
        help="Filename prefix before group key",
    )
    p.add_argument(
        "--all-timepoints",
        action="store_true",
        help="Include every JSON time (day 0, half days, and days beyond 15). Default is image days only.",
    )
    p.add_argument(
        "--min-image-day",
        type=int,
        default=_DEFAULT_IMAGE_DAY_MIN,
        metavar="D",
        help=f"First full day to keep when filtering (default {_DEFAULT_IMAGE_DAY_MIN})",
    )
    p.add_argument(
        "--max-image-day",
        type=int,
        default=_DEFAULT_IMAGE_DAY_MAX,
        metavar="D",
        help=f"Last full day to keep when filtering (default {_DEFAULT_IMAGE_DAY_MAX})",
    )
    p.add_argument(
        "--include-half-day-timepoints",
        action="store_true",
        help="When filtering, allow 0.5, 1.5, ... within the day range (default: integer days only)",
    )
    args = p.parse_args()
    props = (
        [x.strip() for x in args.properties.split(",") if x.strip()]
        if args.properties
        else discover_property_names(args.label_dir, args.file_prefix)
    )
    if not props:
        p.error(
            f"No properties found under {args.label_dir} "
            f"(expected files like {args.file_prefix}_<group>.SEEDS.<PROP>.json)"
        )
    write_labels_csv(
        args.out,
        args.label_dir,
        props,
        reference_encoded_csv=args.reference_encoded_csv,
        file_prefix=args.file_prefix,
        image_timepoints_only=not args.all_timepoints,
        image_day_min=args.min_image_day,
        image_day_max=args.max_image_day,
        allow_half_day_timepoints=args.include_half_day_timepoints,
    )
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
