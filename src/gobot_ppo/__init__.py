from .config import PPOConfig
from .env import GobotCartPoleTargetEnv, GobotPPOEnv, VectorSpace, space_from_spec
from .lqr import CartPoleLQRController, DEFAULT_CARTPOLE_LQR_GAIN, cartpole_lqr_action

__all__ = [
    "CartPoleLQRController",
    "DEFAULT_CARTPOLE_LQR_GAIN",
    "GobotCartPoleTargetEnv",
    "GobotPPOEnv",
    "PPORunner",
    "PPOConfig",
    "VectorSpace",
    "cartpole_lqr_action",
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
    if name in {"PPORunner", "train"}:
        from . import runner as runner_module

        return getattr(runner_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
