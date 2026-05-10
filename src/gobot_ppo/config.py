import json
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class PPOConfig:
    total_steps: int = 4096
    num_envs: int = 1
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
    min_log_std: float = -5.0
    max_log_std: float = 2.0
    action_transform: str = "tanh"
    action_scale: float = 0.25
    action_rate_limit: float = 0.05
    normalize_observations: bool = True
    finite_observation_limit: float = 1.0e6
    finite_reward_limit: float = 1.0e6
    invalid_reward: float = -1.0
    reward_scale: float = 0.01
    log_path: str = "runs/train.csv"
    save_dir: str = "checkpoints"
    save_every: int = 0
    resume: str = ""


def load_config_file(path):
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError("YAML config files require PyYAML. Run `uv sync`.") from error
        data = yaml.safe_load(text)
        return data or {}
    if config_path.suffix.lower() == ".json":
        return json.loads(text)
    raise ValueError(f"unsupported config file format: {config_path.suffix}")


def update_dataclass(instance, values):
    names = {field.name for field in fields(instance)}
    for name, value in values.items():
        if name in names:
            setattr(instance, name, value)
    return instance
