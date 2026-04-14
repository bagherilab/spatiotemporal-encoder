#!/usr/bin/env python3
import argparse
from dataclasses import asdict

import torch

from simulation_encoder.dataclass.param_sets import ModelParams
from simulation_encoder.models.ae import AE
from simulation_encoder.models.vae import VAE
from simulation_encoder.utils.yaml_utils import load_dataset_yaml, load_model_yaml


def create_model_from_config(
    model_name: str,
    dataset_name: str,
    latent_dim: int,
    num_timepoints: int,
    num_epochs: int,
) -> tuple[str, object]:
    dataset_cfg = load_dataset_yaml(dataset_name)
    model_cfg = load_model_yaml(model_name)

    model_params = ModelParams(
        name=model_name,
        model_type=model_cfg.type,  # type: ignore[arg-type]
        architecture=model_cfg.architecture.model_dump(exclude_none=True),  # type: ignore[union-attr]
        num_channels=len(dataset_cfg.channels),
        num_timepoints=num_timepoints,
        num_epochs=num_epochs,
        params={
            "latent_dim": latent_dim,
            "optimizer": {"type": torch.optim.Adam, "lr": 1e-3},
        },
    )
    kwargs = asdict(model_params)
    model_type = kwargs.pop("model_type")

    if model_type == "AE":
        return model_type, AE(**kwargs)
    if model_type == "VAE":
        return model_type, VAE(**kwargs)
    raise ValueError(f"Unsupported model type: {model_type}")


def count_trainable(module: object) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)  # type: ignore[attr-defined]


def format_row(cols: list[str], widths: list[int]) -> str:
    return "  ".join(col.ljust(width) for col, width in zip(cols, widths))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print trainable parameter counts for selected model configs."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ae_small", "cae_small", "neuralop_small", "vit_small"],
        help="Model config names from src/conf/models (without .yaml).",
    )
    parser.add_argument(
        "--dataset",
        default="gastruloid_128",
        help="Dataset config name from src/conf/datasets (without .yaml).",
    )
    parser.add_argument("--latent-dim", type=int, default=16, help="latent_dim override.")
    parser.add_argument(
        "--num-timepoints", type=int, default=9, help="num_timepoints override."
    )
    parser.add_argument("--num-epochs", type=int, default=1, help="num_epochs placeholder.")
    args = parser.parse_args()

    headers = ["model", "type", "total", "encoder", "decoder_image", "decoder_timepoint"]
    rows: list[list[str]] = []

    for model_name in args.models:
        model_type, model = create_model_from_config(
            model_name=model_name,
            dataset_name=args.dataset,
            latent_dim=args.latent_dim,
            num_timepoints=args.num_timepoints,
            num_epochs=args.num_epochs,
        )
        total = count_trainable(model)
        encoder = count_trainable(model.encoder)  # type: ignore[attr-defined]
        decoder_image = count_trainable(model.decoder_image)  # type: ignore[attr-defined]
        decoder_timepoint = count_trainable(model.decoder_timepoint)  # type: ignore[attr-defined]
        rows.append(
            [
                model_name,
                model_type,
                f"{total:,}",
                f"{encoder:,}",
                f"{decoder_image:,}",
                f"{decoder_timepoint:,}",
            ]
        )

    widths = [
        max(len(h), max(len(row[i]) for row in rows))
        for i, h in enumerate(headers)
    ]
    print(
        f"dataset={args.dataset} latent_dim={args.latent_dim} "
        f"num_timepoints={args.num_timepoints}"
    )
    print(format_row(headers, widths))
    print(format_row(["-" * w for w in widths], widths))
    for row in rows:
        print(format_row(row, widths))


if __name__ == "__main__":
    main()
