# Implementation and Attribution Notes

This is a transparent release of the current implementation, not a claim that
all experimental attribution questions have been resolved. Packaging did not
retrain full experiments or impose an algorithm ranking.

## Reward Semantics

`rewards/v2x.py` is the shared source for V2X reward computation. In `resource`
mode, the reward combines a constant shift, bounded latency utility, deadline
margin utility, peak-link power utility, packet-balance utility, delay
improvement bonus, worsening-delay cost, risk severity and memory-guidance
bonus. It is not simply negative latency. Other historical reward modes remain
available for compatibility.

The common YAML explicitly selects the current CLI coefficients; the fallback
constants inside the reward helper differ from some parser defaults. Always
pass a resolved preset instead of interpreting helper defaults as a protocol.
The `v2x_reward_scale` argument is not a multiplier in the `resource` branch.

The guidance bonus aligns actions with a hand-defined SINR/load resource prior.
Retrieved memory tags, backend identity and latency priority weight the bonus.
It does not parse arbitrary `good_action` text into an action evaluator and does
not run an online LLM at each action. A Qwen-specific factor occurs in the
weighting. This must not be described as an independently validated causal
score supplied directly by the LLM.

The memory embedding is a signed token-hash vector, not a pretrained semantic
encoder. Four handcrafted V2X bootstrap records are inserted for memory-enabled
methods. Qwen writes subsequent structured summaries; its prompt requests a
`cause` field but does not establish causality by intervention.

## Method Attribution

| Variant | Retrieved memory | Handcrafted action refinement | Auxiliary risk term |
| --- | --- | --- | --- |
| MAPPO | No | No | No |
| PriorOnly | No | Yes | No |
| Qwen-memory-only | Bootstrap + Qwen | No | No |
| Template-memory-only | Bootstrap + template | No | No |
| Random-memory | Random embedding, no retrieved records | No | No |
| No-risk | Bootstrap + writer | Yes | No |
| Hippo-full | Bootstrap + writer | Yes | Yes, after warmup |

The five-way runner matches supplied seeds, horizons and resource coefficients
and supports the same external reporting window. Enabled reward terms still
differ: no retrieved records means zero guidance bonus. Full also multiplies
the configured action-refinement strength by 1.15, while PriorOnly/no-risk use
1.0. Thus the full/PriorOnly comparison is not an isolated test of an LLM. A
clean factorial ablation would require separate design and new experiments.

The risk head is an auxiliary severity predictor with BCE training and a
detached risk adjustment in PPO; it is not a learned discounted cost-return
critic. The separate MAPPO-Lagrangian runner implements a cost-return critic.

## Baselines and Hyperparameters

VDN/QMIX/IQL use discrete macro-actions mapped onto the hybrid environment;
MAPPO uses continuous power ratios and categorical packet choices. This is
not identical action-space capacity. Macro-actions include domain-informed
resource allocations. Document this when comparing results.

The current **V2X-IPPO** branch instantiates `V2XMAPPO` and evaluates the critic
on the global state. Preserve its historical label for tracing existing runs,
but do not claim it is an independent/local-critic IPPO implementation. The
VMAS branch uses a different critic input. Correcting V2X IPPO requires a
separate algorithm change and reevaluation.

The main V2X runner does not forward its `--hidden` parser option into
`TrainConfig`; the effective width is the dataclass default (128). The value
baseline preset uses width 192. This inherited behavior is preserved and not
presented as a parameter-matched comparison. The full actor also retains an
unused legacy `net` submodule in its parameter set.

## Metric Interpretation

- Reported V2X delay is raw simulator delay multiplied by
  `v2x_delay_report_scale` (historically 0.5). This also enters reward/risk
  computation; it is not merely a display-unit conversion.
- `v2x_risk` is `clip((delay - deadline) / deadline, 0, 1)`, an exceedance
  severity score, not a binary violation frequency. The main episode log
  applies it to mean episode delay; baseline logs average per-step severity.
  These summaries are not directly identical metrics.
- The original `Summary.mean_delay` and `mean_violation` cover the whole run.
  Only its `final_reward` uses a tail window. New final-window reports are
  computed from `EpisodeMetrics`, without relabeling all-run averages.
- `episode_reward` is a mean per-step reward in the full runners, not the
  undiscounted episode sum used by every RL library.
- The optional aligned-start point evaluates a shared neutral action, not the
  initialized neural policies. Release presets turn it off and reports exclude
  it. Episode zero otherwise is a real training episode and is retained.
- The main and baseline runners export no best-policy checkpoint evaluation.
  A best-model latency CDF needs independent checkpoint selection and held-out
  rollout samples; it cannot be obtained from reward sign changes or training
  episode means. No such CDF is included in this release.

The preserved simulator also contains task-specific simplifications. Ownership,
physics calibration, deadline/report scales, action refinement and evaluation
definitions should be reviewed before presenting the release as a final,
fully validated benchmark.
