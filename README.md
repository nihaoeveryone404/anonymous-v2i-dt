# Anonymous V2I Digital Twin Code

Authors: Anonymous

## Overview

This repository contains the anonymous implementation of a CPFL-enabled V2I
digital twin framework with personalized PL/KPI prediction and DT-assisted V2I
association. It provides dataset preparation, personalized federated learning,
map inference, association evaluation, communication accounting, and plotting
code. Data, checkpoints, logs, generated arrays, figures, and experiment results
are intentionally not distributed.

## Repository Structure

- `configs/`: YAML configurations for every pipeline stage.
- `src/data/`: RadioMapSeer preparation, Non-IID partitioning, feature and label generation.
- `src/models/`: shared CPFL model definitions.
- `src/training/`: PL/KPI CPFL training, personalization, compression, and evaluation.
- `src/inference/`: PL/KPI prediction and map export.
- `src/decision/`: DT-assisted association, sensitivity analysis, metrics, and visualization.
- `src/utils/`: configuration, I/O, metric, wireless, and plotting helpers.
- `scripts/`: shell wrappers for the complete pipeline.
- `data/`: dataset placement instructions only.
- `outputs/`: runtime-output instructions only; generated content is ignored.
- `docs/`: reproduction details and items requiring manual scientific review.

## Environment

Python 3.9 is the reference version. From the repository root:

```bash
conda create -n v2idt python=3.9
conda activate v2idt
pip install -r requirements.txt
```

Alternatively, create the complete environment with:

```bash
conda env create -f environment.yml
conda activate v2idt
```

The pinned `torch` package is platform-neutral. CUDA users may install the
appropriate PyTorch build for their driver before installing the remaining
requirements. Training automatically uses CUDA when it is available unless a
different device is supplied on the command line.

## Dataset

The original RadioMapSeer dataset is not redistributed in this repository.

Please download RadioMapSeer from its official source and organize it as:

```text
data/
└── RadioMapSeer/
    ├── gain/
    ├── antenna/
    ├── png/
    └── polygon/
```

The preparation code validates the files it consumes. Depending on the
RadioMapSeer release, aliases for propagation-model directory names are handled
by `dataset_preparation.py`.

## Configuration

All paths, task/scenario switches, training settings, wireless parameters, and
association sensitivity values are stored in `configs/*.yaml`. Paths are
relative to the repository root. Explicit command-line options override YAML
values; for example:

```bash
python -m src.training.train_cpfl --config configs/train_pl.yaml --rounds 10
```

Review each configuration before a full run, especially the scenario, number
of rounds, checkpoint path, and optional personalized-head directory.

## Step 1: Dataset preparation

```bash
python -m src.data.dataset_preparation \
    --config configs/dataset.yaml
```

## Step 2: Train PL model

```bash
python -m src.training.train_cpfl \
    --config configs/train_pl.yaml
```

## Step 3: Train KPI model

```bash
python -m src.training.train_cpfl \
    --config configs/train_kpi.yaml
```

## Step 4: Prediction

```bash
python -m src.inference.predict \
    --config configs/predict_pl.yaml

python -m src.inference.predict \
    --config configs/predict_kpi.yaml
```

The supplied prediction configurations expect the final round-400 checkpoints
created by the supplied training configurations. If the round count changes,
update `model_ckpt` and, when applicable, `local_head_dir` in the prediction
configuration.

## Step 5: DT-assisted association

```bash
python -m src.decision.decision \
    --config configs/decision.yaml
```

Generated metrics and visualizations are written below `outputs/` and remain
excluded from Git. The plotting capability is retained, but no generated figure
is included in this release.

## Reproduction Notes

See [`docs/reproduction.md`](docs/reproduction.md) for expected stage outputs,
configuration precedence, and scientific-definition items preserved from the
original research scripts for manual confirmation.

## Citation

Citation information will be released after publication.
