from .config import PPOConfig
from .env import GobotBox, GobotGymEnv, add_gobot_pythonpath, space_from_spec
from .runner import PPORunner, train

__all__ = [
    "GobotBox",
    "GobotGymEnv",
    "PPOConfig",
    "PPORunner",
    "add_gobot_pythonpath",
    "evaluate",
    "load_policy",
    "policy_action",
    "space_from_spec",
    "train",
]


def __getattr__(name):
    if name in {"evaluate", "load_policy", "policy_action"}:
        from . import eval as eval_module

        return getattr(eval_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
