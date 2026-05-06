import argparse

from .config import PPOConfig, load_config_file, update_dataclass
from .runner import train


def _section(config, name):
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def _pick(cli_value, config, name, default=None):
    if cli_value is not None:
        return cli_value
    return config.get(name, default)


def main():
    parser = argparse.ArgumentParser(description="Train a single Gobot RL environment with PPO.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--gobot-pythonpath", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--robot", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--update-epochs", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--initial-log-std", type=float, default=None)
    parser.add_argument("--action-scale", type=float, default=None)
    parser.add_argument("--action-rate-limit", type=float, default=None)
    parser.add_argument("--finite-observation-limit", type=float, default=None)
    parser.add_argument("--finite-reward-limit", type=float, default=None)
    parser.add_argument("--invalid-reward", type=float, default=None)
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    file_config = load_config_file(args.config) if args.config else {}
    env_config = _section(file_config, "env")
    ppo_values = _section(file_config, "ppo")
    config = update_dataclass(PPOConfig(), ppo_values)
    cli_overrides = {
        "total_steps": args.total_steps,
        "rollout_steps": args.rollout_steps,
        "update_epochs": args.update_epochs,
        "minibatch_size": args.minibatch_size,
        "learning_rate": args.learning_rate,
        "initial_log_std": args.initial_log_std,
        "action_scale": args.action_scale,
        "action_rate_limit": args.action_rate_limit,
        "finite_observation_limit": args.finite_observation_limit,
        "finite_reward_limit": args.finite_reward_limit,
        "invalid_reward": args.invalid_reward,
        "log_path": args.log_path,
        "save_dir": args.save_dir,
        "save_every": args.save_every,
        "resume": args.resume,
        "seed": args.seed,
    }
    update_dataclass(config, {name: value for name, value in cli_overrides.items() if value is not None})

    result = train(
        scene_path=_pick(args.scene, env_config, "scene", ""),
        robot=_pick(args.robot, env_config, "robot", "robot"),
        backend=_pick(args.backend, env_config, "backend", "null"),
        project_path=_pick(args.project, env_config, "project", None),
        config=config,
        device=_pick(args.device, env_config, "device", "cpu"),
        gobot_pythonpath=_pick(args.gobot_pythonpath, env_config, "gobot_pythonpath", None),
    )
    print(result)


if __name__ == "__main__":
    main()
