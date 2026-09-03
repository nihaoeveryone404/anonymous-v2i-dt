# Anonymization Report

## Scope reviewed

The four supplied research scripts were reviewed and reorganized as dataset
preparation, CPFL training, inference, and decision modules. The review covered
source code, YAML configurations, shell entry points, documentation, dependency
files, and the final archive manifest.

## Identity checks

The repository was scanned case-insensitively for personal names supplied in
the review checklist, institutional names and abbreviations, institutional email
domains, generic email-address patterns, local usernames, and remnants of the
original local project directory names. No author identity, affiliation, or
email address was detected after cleanup. The README identifies the authors only
as `Anonymous`.

## Path cleanup

All local Windows absolute paths were removed from the four scripts. Dataset,
prepared-data, checkpoint, prediction, metric, and visualization locations now
come from YAML files and use repository-relative paths under `data/` and
`outputs/`. Legacy searches for personally named experiment directories were
removed. Explicit CLI arguments can override YAML values.

## Configuration migration

Task, mode, Non-IID scenario, alpha/beta, NLOS confidence, traffic settings,
training rounds, local epochs, batch size, Top-K upload ratio, quantization bits,
and related wireless/evaluation parameters are represented in YAML. Users do
not need to edit Python source to switch the published pipeline stages.

## Excluded data and artifacts

The `.gitignore` excludes the RadioMapSeer dataset, all runtime output below
`outputs/` except its README, checkpoints, generated NumPy arrays, experiment
CSVs and logs, generated PNG/PDF files, Python caches, IDE files, OS metadata,
and temporary directories. The reviewed repository contains none of those
generated or experimental artifacts.

## Engineering changes

The original algorithms were retained. Shared PL/KPI model definitions were
moved to `src/models/cpfl_models.py`; configuration and filesystem helpers,
metrics, wireless conversions, and plotting helpers are organized under
`src/utils/`. The four command-line modules remain responsible for their
original scientific stages.

Potential scientific-definition differences and the personalized-head export
limitation are documented in `docs/reproduction.md` rather than silently
rewritten.

## Validation result

- Python syntax: PASS (`python -m compileall src`)
- Six YAML-backed command invocations with `--help`: PASS
- Windows absolute path scan: PASS (no matches)
- Requested identity/institution/email keyword scan: PASS (no matches)
- Configuration path audit: PASS (all configured paths are relative)
- Checkpoint/data/result artifact scan: PASS (none present)
- README command/module consistency: PASS

No full training or numerical experiment was run because the dataset and
checkpoints are intentionally excluded from this anonymous code release.
