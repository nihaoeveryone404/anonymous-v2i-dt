# Reproduction notes

## Scope

This release is an engineering and anonymization pass over four research
scripts. The data-generation, Non-IID partition, PL/KPI label definitions, LOS
construction, personalized federated aggregation, Top-K error-feedback
compression, quantization, prediction, association, and evaluation formulas
were preserved. No dataset or experimental artifact is bundled.

Run all commands from the repository root. A YAML file is loaded first, after
which explicitly supplied command-line options take precedence.

## Expected runtime artifacts

Dataset preparation writes manifests, PL/KPI user arrays, normalization data,
LOS arrays, and schema metadata under `outputs/dataset/`. Training writes
checkpoints, communication logs, configuration snapshots, and evaluation CSVs
under `outputs/checkpoints/{pl,kpi}/`. Prediction writes arrays, summaries, and
optional plots under `outputs/predictions/`. Decision evaluation writes metrics
and figures under `outputs/decision/`. These artifacts are all ignored by Git.

## Preserved items requiring scientific confirmation

The original scripts use different default association/load regimes in data
label generation and final decision evaluation. In particular, dataset KPI
labels use `alpha=1`, `beta=1`, `conf_nlos=0.5`, requested rate `0.2 Mbps`, and
average file size `2 MB`; decision evaluation uses `alpha=3`, `beta=10`,
`conf_nlos=0.01`, requested rate `20 Mbps`, and average file size `10 MB`.
These values are exposed in separate YAML files and were not reconciled because
doing so would change the experimental definitions.

The training script preserves personalized heads in each in-memory client, but
its original server loop saves global model checkpoints only. Prediction can
optionally load per-client head directories. Consequently, the supplied global
checkpoint path is reproducible, while use of personalized-head directories
requires confirming the intended head-export procedure with the research
authors. This release does not invent such a procedure.

The prediction code also preserves a fallback confidence/LOS construction from
received-power maps when a compatible geometric LOS tensor is unavailable.
This fallback and its precedence relative to prepared LOS tensors are unchanged.

## Lightweight validation

Syntax and configuration wiring can be checked without data or checkpoints:

```bash
python -m compileall src
python -m src.data.dataset_preparation --help
python -m src.training.train_cpfl --help
python -m src.inference.predict --help
python -m src.decision.decision --help
```

A complete numerical reproduction requires the external RadioMapSeer dataset,
substantial training time, and hardware appropriate for the selected settings.
