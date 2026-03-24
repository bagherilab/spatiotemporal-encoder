"""Downstream classification on encoder latents under ``results/<study_name>/``.

- `BACKEND = "point"`: sklearn head on latent vectors at one time point
- `BACKEND = "trajectory"`: RNN/LSTM on the full latent sequence
"""

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from latent_model.runner import (
    classification_dict_to_per_example,
    discover_models_with_best_checkpoint,
    run_study_classification,
    run_study_trajectory_classification,
)


def main() -> None:
    repo = _REPO

    RESULTS_STUDY_DIR = repo / "results" / "architecture-ARCADE"

    # "point" | "trajectory"
    BACKEND = "trajectory"

    # --- point configs ---
    CLASSIFIER_TYPE = "logistic_regression" # "logistic_regression" | "random_forest" | "svm" | "mlp"
    TIME_POINT_IDX = 0
    CV_FOLDS = 5

    # --- trajectory configs ---
    TRAJECTORY_CELL = "lstm"  # "lstm" | "rnn"
    TRAJECTORY_NUM_CLASSES = 2
    TRAJECTORY_MAX_EPOCHS = 50
    TRAJECTORY_PATIENCE = 5

    RANDOM_SEED = 42

    try:
        study_dir = RESULTS_STUDY_DIR.expanduser().resolve()

        if not discover_models_with_best_checkpoint(study_dir):
            raise FileNotFoundError(
                f"No subfolders with _best_model under {study_dir}"
            )

        if BACKEND == "point":
            raw = run_study_classification(
                study_dir,
                time_point_idx=TIME_POINT_IDX,
                classifier_type=CLASSIFIER_TYPE,
                random_seed=RANDOM_SEED,
                cv_folds=CV_FOLDS,
            )
            stem = f"{raw['study_name']}_classification_results"
        elif BACKEND == "trajectory":
            raw = run_study_trajectory_classification(
                study_dir,
                random_seed=RANDOM_SEED,
                cell=TRAJECTORY_CELL,
                num_classes=TRAJECTORY_NUM_CLASSES,
                max_epochs=TRAJECTORY_MAX_EPOCHS,
                patience=TRAJECTORY_PATIENCE,
            )
            stem = f"{raw['study_name']}_trajectory_classification_results"
        else:
            raise ValueError(f"Unknown BACKEND: {BACKEND!r}")

        out_dir = repo / "results_downstream" / "classification_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        cls_path = out_dir / f"{stem}.json"
        with open(cls_path, "w", encoding="utf-8") as f:
            json.dump(raw, f)

        per = classification_dict_to_per_example(raw)
        per["source_file"] = str(cls_path)
        per_path = out_dir / f"{stem}_per_example.json"
        with open(per_path, "w", encoding="utf-8") as f:
            json.dump(per, f)

        n = len(raw["model_dataset_results"])
        print(f"Wrote {cls_path} and {per_path} ({n} model/dataset rows, backend={BACKEND})")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
