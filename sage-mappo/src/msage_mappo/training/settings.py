from dataclasses import dataclass


@dataclass
class TrainConfig:
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    risk_coef: float = 0.25
    lagrangian_cost_limit: float = 0.005
    lagrangian_lr: float = 0.2
    lagrangian_init: float = 0.0
    lagrangian_max: float = 20.0
    lr: float = 3e-4
    ppo_epochs: int = 3
    hidden: int = 128
    memory_dim: int = 64
    ctx_dim: int = 4
    top_k: int = 3
    v2x_deadline_ms: float = 50.0
    v2x_reward_scale: float = 1.0
    vmas_bc_coef: float = 0.05
    vmas_reward_scale: float = 1.0
    vmas_log_std_init: float = -0.4
