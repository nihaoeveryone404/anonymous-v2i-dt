# Reproduction Guide

## Configuration Precedence

Precedence is: built-in parser defaults, inherited YAML files (in listed order),
the selected YAML file, then explicit command-line arguments. `extends` paths
are relative to the YAML file that declares them. Other relative paths,
including `--config`, input, output, model and environment paths, are resolved
from the repository root, not the shell working directory.

Unknown keys, invalid choices/types, nonpositive horizons, duplicate seeds and
cyclic inheritance fail early. Boolean YAML values must be actual booleans.
The code is intended to run from a clone or editable installation, where
repository-level configs and data directories are present.

## Presets

| Preset | Entry point | Purpose |
| --- | --- | --- |
| `toy_smoke.yaml` | `train_toy_v2x.py` | Toy prototype smoke run |
| `v2x_smoke.yaml` | `run_full_experiments.py` | MAPPO/full, template writer, 3 episodes |
| `v2x_baselines_smoke.yaml` | `run_additional_baselines.py` | Four historical baselines, reduced replay threshold |
| `v2x_main.yaml` | `run_full_experiments.py` | Random, MAPPO, no-risk and full |
| `v2x_baselines.yaml` | `run_additional_baselines.py` | IQL, VDN, QMIX and historical V2X-IPPO |
| `v2x_ablation.yaml` | `run_matched_v2x_attribution_ablation.py` | Five attribution variants |
| `v2x_safe_baseline.yaml` | `run_v2x_safe_baseline.py` | MAPPO-Lagrangian |
| `vmas_main.yaml` | `run_full_experiments.py` | Optional VMAS navigation |

`v2x_common.yaml` is an inherited settings file, not a standalone run. Presets
are explicit starting protocols based on the current runner; they are not
claimed to reconstruct every historical paper figure. In particular, reward
weights have changed between historical runs.

## Monitoring and Outputs

Use `python -u` and optionally redirect stdout/stderr to a local log. The V2X
presets print progress every 100 episodes and at each method/seed completion.
Override with `--progress-every 500` if desired. Updates include reward, delay,
risk and losses; value-based methods also report replay/update counters.

Main and baseline runners write the workbook at the end of the whole command:

| Sheet | Contents |
| --- | --- |
| `EpisodeMetrics` | Recorded training metrics for each method/seed/episode |
| `StepMetrics` | Environment and reward-component diagnostics |
| `SemanticMemory` | Writer provenance and memory summaries; main runner only |
| `Config` | Resolved arguments and paths |
| `Summary` | Historical aggregate conventions; see protocol notes |

Do not interrupt a long run expecting checkpoint-based resume: these main
runners do not implement it. The safe runner writes incremental JSONL and
supports skipping completed seeds, not restarting mid-seed. Generated local
configs/logs may contain local paths, so inspect them before sharing.

## Reporting

`plot_results.py` reads one or more raw workbooks and rejects duplicate
method/seed/episode records. It keeps scenario identifiers and filenames,
excludes explicitly marked common-action diagnostics, and does not merge
different scenarios into one estimate.

Final-window reporting takes the last N recorded training episodes separately
for each method, scenario and seed, then reports the equally weighted mean and
sample standard deviation across seeds. Short runs show the actual number of
episodes used. The reporter preserves the historical metric definition; it
does not silently convert a risk severity score into a violation probability.

Smoothing is a trailing mean performed separately within each seed, used only
for plots. `--smooth-window 1` is unsmoothed. Shading, where multiple seeds are
available at every point, is one standard deviation of the plotted seed
series, not a confidence interval. A single seed produces no band.

The output `report_protocol.json` records input SHA-256 checksums, window and
smoothing settings. `episode_metrics.csv` preserves numeric samples;
`plot_points.csv` contains the plotted statistics. No offsets, ranking changes,
curve stretching, fabricated starting points or CDF synthesis are performed.

## Before Reporting Paper Results

Record exact versions of Python/PyTorch/NumPy, GPU hardware, environment hash,
all CLI overrides, Qwen weight hash/source, actual writer-backend records and
the reporting window. Training is stochastic; CPU and GPU results need not
be bit-identical. Read the limitations below before interpreting comparisons.
Smoke-test success verifies execution only, not convergence or superiority.
