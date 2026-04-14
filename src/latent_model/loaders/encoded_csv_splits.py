"""Train/val/test sample ids from ``encoded_data.csv`` ``split`` column."""

from pathlib import Path

import pandas as pd


def train_val_test_ids_from_encoded_csv(
    data: str | Path | pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    """
    Return sorted unique ``sample_id`` lists for each split in the encoder CSV.
    """
    if isinstance(data, (str, Path)):
        df = pd.read_csv(data)
    else:
        df = data

    if "split" not in df.columns:
        raise ValueError("encoded_data.csv must contain a 'split' column")
    if "sample_id" not in df.columns:
        raise ValueError("encoded_data.csv must contain a 'sample_id' column")

    work = df[["sample_id", "split"]].copy()
    work["sample_id"] = work["sample_id"].astype(str).str.strip()
    work["split"] = work["split"].astype(str).str.strip().str.lower()

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    for sid, grp in work.groupby("sample_id", sort=False):
        splits = grp["split"].unique()
        raw_sp = splits[0] if len(splits) else None
        if raw_sp is None or (isinstance(raw_sp, float) and pd.isna(raw_sp)):
            raise ValueError(f"encoded_data.csv: missing split for sample_id={sid!r}")
        sp = str(raw_sp).strip().lower()
        if sp in ("train", "training", "tr"):
            train_ids.append(sid)
        elif sp in ("val", "validation", "valid", "dev"):
            val_ids.append(sid)
        elif sp in ("test", "testing", "te", "holdout"):
            test_ids.append(sid)
        else:
            raise ValueError(
                f"encoded_data.csv: unknown split {sp!r} for sample_id={sid!r} "
                "(expected train / val / test)"
            )

    train_ids.sort()
    val_ids.sort()
    test_ids.sort()
    return train_ids, val_ids, test_ids
