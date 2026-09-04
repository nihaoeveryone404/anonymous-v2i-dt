# MSAGE-MAPPO

Code-only research implementation of semantic-memory-assisted multi-agent
reinforcement learning for V2X resource allocation, with an optional VMAS
navigation benchmark. Historical implementation identifiers use `M-SAGE` and
`msage_mappo`; these names are retained to keep experiment records traceable.

The repository includes a hybrid power/packet actor, a centralized return and
risk critic, Qwen/template trajectory-to-memory writers, memory retrieval,
resource-reward shaping, and comparison runners. It does **not** distribute
datasets, checkpoints, pretrained LLM weights, logs, or paper result figures.

## Repository Layout

```text
configs/                 Shared YAML settings and run presets
data/                    Local inputs and separately obtained model weights
docs/                    Reproduction, implementation notes and provenance
outputs/                 Generated artifacts (ignored by Git)
scripts/                 Short command-line entry points
src/msage_mappo/
  envs/                  Full V2X simulator and toy smoke-test environment
  models/                V2X/VMAS networks, value networks and QMIX mixer
  rewards/               V2X resource reward and memory-guidance bonus
  training/              Main method, baselines, safe baseline and ablations
  evaluation/            Raw-metric summaries and learning curves
  utils/                 Validated configuration and repository-relative paths
  memory.py              Semantic records and constraint-aware retrieval
  llm_writer.py          Local Qwen interface and explicit template backend
  embedding.py           Lightweight hashed text embedding
  agent.py               Toy-only PPO implementation
tests/                   Configuration, source preservation and behavior tests
```

See [the code map](docs/code_map.md) for the exact entry points and
[tree.txt](tree.txt) for the complete file list.

## Installation

Python 3.11 is used for local validation. Run in an isolated environment:

```bash
conda env create -f environment.yml
conda activate hippo-mappo
python -m pip install -e . --no-deps
```

Alternatively, use an existing Python virtual environment and install
`requirements.txt`, then install the project with the editable command above.
For GPU use, choose a PyTorch build compatible with your CUDA installation.
Optional packages are separate:

```bash
python -m pip install -r requirements-llm.txt
python -m pip install -r requirements-vmas.txt
```

Install only the optional backend you need. The GGUF model is obtained
separately; see [data/models/README.md](data/models/README.md).

## Quick Start

These commands run actual short CPU training jobs without downloading an LLM:

```bash
python -m unittest discover -s tests -v
python scripts/train_toy_v2x.py --config configs/toy_smoke.yaml
python scripts/run_full_experiments.py --config configs/v2x_smoke.yaml
python scripts/run_additional_baselines.py --config configs/v2x_baselines_smoke.yaml
python scripts/plot_results.py --input outputs/smoke/main.xlsx outputs/smoke/baselines.xlsx --output-dir outputs/smoke/report --last-window 2
```

Existing training outputs are not overwritten. Choose another `--output` or
`--output-dir` when repeating a run. Smoke tests check execution and updates;
they are not convergence studies or paper results.

## Experiments

The V2X presets use 1,000 episodes, eight steps per episode and five seeds. Main
and baseline presets inherit the same resource-reward coefficients:

```bash
python -u scripts/run_full_experiments.py --config configs/v2x_main.yaml
python -u scripts/run_additional_baselines.py --config configs/v2x_baselines.yaml
python -u scripts/run_matched_v2x_attribution_ablation.py --config configs/v2x_ablation.yaml
```

Main/ablation presets require Qwen weights. Use `--model-path` to point to them;
use `--device cuda` and `--llm-gpu-layers` only when supported by your installed
backends. Selecting `--llm-backend template` is a different experiment, not an
LLM result. The five-way ablation explicitly disallows template fallback for
its Qwen group. Inspect its commands first with `--dry-run`.

The inherited `ippo` V2X implementation still uses a centralized critic. Its
historical name is preserved, but it is **not a strict IPPO reference**. See
[implementation limitations](docs/protocol_notes.md) before using any result
to support algorithm attribution. No ranking or convergence outcome is imposed.

## Results and Reproduction

```bash
python scripts/plot_results.py --input outputs/v2x_main/results.xlsx outputs/v2x_baselines/results.xlsx --output-dir outputs/v2x_report --last-window 100 --smooth-window 25
```

This creates raw episode CSV, per-seed final-window statistics, aggregate
statistics, plotted points, and PNG/PDF learning curves. Display smoothing is
explicit, rewards/delays are not shifted, and single-seed runs receive no
artificial uncertainty band. Common-action initial diagnostics are excluded.
Training curves are not best-checkpoint latency CDFs.

- [Reproduction guide](docs/reproduction.md)
- [Reward, baseline and attribution limitations](docs/protocol_notes.md)
- [Local verification record](docs/validation.md)
- [Publishing instructions](docs/github_upload.md)
- [Anonymization and exclusions](ANONYMIZATION_REPORT.md)

## License and Citation

Licensing remains a release decision for the copyright holder. `LICENSE` is a
clearly marked pending notice, not an open-source license grant. Confirm source
ownership and replace it before public distribution. Add the final paper
citation after its publication details are settled.
