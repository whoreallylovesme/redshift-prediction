# Gemini Project Context: Astro

This repository is a machine learning project dedicated to **photometric redshift (photoz) prediction**. It provides a flexible, configuration-driven pipeline for both tabular data and deep learning (NN) models.

## Project Overview

- **Core Task:** Predicting photometric redshifts (`photoz`) from galaxy features and images.
- **Key Dataset:** SDSS/Teddy benchmark (A, B, C, D splits), where D represents the most challenging OOD (Out-Of-Distribution) scenario.
- **Architecture:** 
    - `lib/`: Core library logic, metrics, and NN model implementations.
    - `runtime/`: CLI entry points for individual pipeline stages.
    - `flows/`: Prefect-orchestrated pipelines.
    - `configs/`: YAML-based configurations for data, models, and flows.
- **Primary Technologies:** Python 3.13, `uv`, `torch`, `catboost`, `scikit-learn`, `astropy`, `prefect`.

## Development Standards

- **Python Version:** ≥ 3.13 (managed via `uv`).
- **Type Safety:** Strict `mypy` configuration. All functions must be fully type-annotated.
- **Linting & Formatting:** `ruff` with a 100-character line limit. Rules: `E`, `F`, `I`, `B`, `UP`.
- **Validation:** No unit tests yet; validation is performed via smoke-test pipelines (e.g., `configs/flows/fast_test.yaml`).

## Key Commands

### Setup & Environment
```bash
uv sync                      # Install dependencies
uv run pre-commit install    # Install git hooks
```

### Linting & Type Checking
```bash
uv run ruff check .          # Run linter
uv run mypy .                # Run type checker
```

### Running Pipelines (Prefect)
```bash
# Smoke test (fast)
uv run python flows/pipeline.py --config configs/flows/fast_test.yaml

# Full tabular pipeline
uv run python flows/pipeline.py --config configs/flows/run_all.yaml

# Full NN pipeline (Teddy A -> B/C/D)
uv run python flows/pipeline.py --config configs/flows/train_nn.yaml
```

### Individual Stages (CLI)
Each script in `runtime/` is a standalone Click CLI.
```bash
uv run python runtime/prepare_dataset.py --help
uv run python runtime/train.py --config configs/tabular/catboost.yaml --outdir runs/catboost_test
uv run python runtime/inference.py --help
uv run python runtime/calc_metrics.py --help
```

## Configuration-Driven Design

The project relies heavily on YAML configs:
1.  **Tabular Configs** (`configs/tabular/`): Define the `estimator` (Python code string) and `param_grid` for scikit-learn pipelines.
2.  **NN Configs** (`configs/nn/`): Define model architecture (often `timm`/`torchvision` backbones), data augmentation, and training hyperparameters.
3.  **Flow Configs** (`configs/flows/`): Orchestrate the pipeline mode (`tabular` vs `nn`) and model list.

## Domain-Specific Metrics (`lib/metrics.py`)

Evaluation uses specialized astronomical metrics:
- `photoz_dz`: $(z_{pred} - z_{true}) / (1 + z_{true})$
- `photoz_bias`: Median of `dz`.
- `photoz_nmad`: Normalized Median Absolute Deviation of `dz`.
- `outlier_fraction`: Percentage of predictions where $|dz| > 0.15$.

## Directory Structure

- `lib/`: Reusable components (metrics, plotting, training loops).
- `lib/nn/`: Neural network specific code (dataloaders, PyTorch utils).
- `runtime/`: CLI scripts for data prep, training, and evaluation.
- `flows/`: Prefect workflow definitions.
- `configs/`: YAML configs organized by purpose.
- `features/`: Definitions of feature sets (e.g., `set1`, `set2`).
- `runs/`: Output directory for models and results (ignored by git).
- `prepared/`: Intermediate Parquet data (ignored by git).
- `datasets/`: Raw data storage (ignored by git).
