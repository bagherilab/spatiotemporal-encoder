# Data directory

Generated and analysis data files live here so the project root stays clean.

- **`csv/`** – CSV outputs (embeddings, metadata, heatmaps, etc.). Pipeline-encoded data is written under `results/` instead.
- **`figures/`** – Plot and figure outputs (e.g. swarm plots, heatmaps).

Contents are ignored by git (`data/*` in `.gitignore`). Point scripts or notebooks at `data/csv/` or `data/figures/` when reading/writing these files.
