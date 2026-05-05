from dataclasses import dataclass


@dataclass
class PPOConfig:
    total_steps: int = 4096
    rollout_steps: int = 256
    update_epochs: int = 4
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    seed: int = 1
    hidden_size: int = 128
    initial_log_std: float = -1.0
