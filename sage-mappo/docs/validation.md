# Local Verification

Date: 2026-09-04. Scope: code packaging and short CPU execution, not a rerun of
the paper's 1,000-episode experiments.

## Environment

Validation used an isolated Python 3.11 virtual environment with access to the
existing scientific packages. The working project's environment was not
upgraded. The release was installed with `pip install -e . --no-deps
--no-build-isolation`.

| Package | Locally tested version |
| --- | --- |
| Python | 3.11.15 |
| PyTorch | 2.6.0+cu124, CPU execution |
| NumPy | 2.4.6 |
| pandas | 3.0.3 |
| openpyxl | 3.1.5 |
| Matplotlib | 3.10.9 |
| PyYAML | 6.0.3 |

Requirement ranges are compatibility targets, not a bit-for-bit lockfile.

## Executed Checks

- 12 unit tests: preset parsing, YAML inheritance/validation/cycles, CLI
  precedence, common reward settings, deterministic environment reset/step,
  packet budget, reward components, memory retrieval, explicit missing-Qwen
  failure, QMIX monotonicity, group-config round trips and reporting semantics.
- All 13 extracted definitions match their source AST fingerprints.
- Editable installation and command-line entry points.
- Toy smoke: 3 episodes, 4 steps, template writer, one seed.
- Main V2X smoke: MAPPO and full, 3 episodes x 3 steps each, one seed,
  template writer, risk updates enabled from the first episode.
- Baseline smoke: IQL, VDN, QMIX and historical V2X-IPPO, 3 episodes x 3 steps,
  one seed. Replay/update counters and nonempty losses were observed.
- MAPPO-Lagrangian smoke: 2 episodes x 3 steps, one seed, incremental logs.
- Raw-workbook reporting: 18 episode records across the six main/baseline
  labels, final-two-episode summaries and PNG/PDF curves.
- Main and baseline commands launched from outside the repository to check
  repository-relative config and simulator paths.
- Five-way attribution `--dry-run`: shared configs and method/backend mapping.
- Publication-file scan and code-only archive integrity check.

Short-run value curves may coincide during nearly all-random epsilon
exploration. No points were changed to separate those curves. Finite loss and
optimizer activity demonstrate execution, not convergence or superiority.

## Not Executed

- Full 1,000-episode or multi-seed paper experiments.
- Qwen inference or its full five-way training ablation.
- VMAS training, remote GitHub Actions, or a clean Linux install.
- Independent best-checkpoint selection or latency-CDF evaluation.

Smoke artifacts were stored outside the release directory. No local smoke
metrics, images or model weights are included in the code-only archive.
