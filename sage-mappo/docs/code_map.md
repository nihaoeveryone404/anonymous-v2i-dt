# Code Map

## Main Implementation

| Component | Authoritative file in this release |
| --- | --- |
| Full V2X environment | `src/msage_mappo/envs/v2x.py` |
| Hybrid actor, return critic, risk critic | `src/msage_mappo/models/v2x.py` |
| VMAS networks | `src/msage_mappo/models/vmas.py` |
| IQL/VDN/QMIX value network and mixer | `src/msage_mappo/models/value_decomposition.py` |
| Resource reward and memory bonus | `src/msage_mappo/rewards/v2x.py` |
| PPO updates, action prior, rollouts | `src/msage_mappo/training/full.py` |
| Value baselines and historical IPPO runner | `src/msage_mappo/training/baselines.py` |
| MAPPO-Lagrangian comparison | `src/msage_mappo/training/safe_baseline.py` |
| Five-way attribution orchestration | `src/msage_mappo/training/matched_ablation.py` |
| Record schema, retrieval and persistence | `src/msage_mappo/memory.py` |
| Qwen prompt, JSON parsing, templates | `src/msage_mappo/llm_writer.py` |
| Text embedding | `src/msage_mappo/embedding.py` |
| Training-curve reports | `src/msage_mappo/evaluation/report.py` |

`scripts/` contains small launchers; model and training definitions are in
`src/`. The toy environment, `agent.py`, and `training/toy.py` are an independent
smoke-test prototype, not the full V2X algorithm used by the main runner.

## Source Preservation

This release was extracted from the working project's `scripts/` and
`src/msage_mappo/`. The full V2X simulator came from
`submission_package/external/v2x_env/new.py`; its source hash matched the
environment referenced by the working runner at packaging time. Only its
`set_seed` and `VehicleToBSEnv` definitions were retained; its separate legacy
trainer and plotting driver were not copied.

`source_manifest.json` records source-file SHA-256 values and AST fingerprints
for 13 extracted model, environment, configuration, and reward definitions.
Unit tests verify those definitions remain equivalent after relocation.

## Release-Specific Changes

1. Internal imports now refer to package modules; external simulator/model
   paths were replaced by repository-relative defaults.
2. YAML inheritance, type/choice validation, explicit CLI overrides and
   configurable PyTorch CPU threads were added.
3. Qwen fallback is opt-in. The requested/actual backend is recorded, and each
   generated memory retains its writer backend.
4. V2X-only baseline execution no longer imports VMAS unconditionally.
5. Training launchers reject an existing target workbook. The safe runner
   accepts a V2X-only `domains` field to share the common config.
6. The ablation orchestrator forwards all resolved parameters, retains raw
   group files, and adds traceable display names without changing numbers.
7. Reports recompute final-window statistics from raw `EpisodeMetrics`; no
   processed paper figures or ranking-adjustment scripts are reused.

The numerical reward, simulator equations, PPO update, retrieval scoring,
action-prior strength, baseline action mappings and historical metric formulas
were not tuned during packaging. Presets explicitly disable the historical
common-action initial diagnostic, and smoke presets use shorter horizons.
