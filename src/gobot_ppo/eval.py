import argparse
from pathlib import Path

from .config import PPOConfig, load_config_file, update_dataclass
from .env import GobotGymEnv, add_gobot_pythonpath
from .runner import ActorCritic, _require_torch


def _section(config, name):
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def _pick(cli_value, config, name, default=None):
    if cli_value is not None:
        return cli_value
    return config.get(name, default)


def load_policy(checkpoint_path, observation_size, action_size, config=None, device="cpu"):
    torch, _ = _require_torch()
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config", {})
    hidden_size = int(checkpoint_config.get("hidden_size", getattr(config, "hidden_size", 128)))
    initial_log_std = float(
        checkpoint_config.get("initial_log_std", getattr(config, "initial_log_std", -1.0))
    )
    min_log_std = float(checkpoint_config.get("min_log_std", getattr(config, "min_log_std", -5.0)))
    max_log_std = float(checkpoint_config.get("max_log_std", getattr(config, "max_log_std", 2.0)))
    model = ActorCritic(
        observation_size,
        action_size,
        hidden_size=hidden_size,
        initial_log_std=initial_log_std,
        min_log_std=min_log_std,
        max_log_std=max_log_std,
    ).model.to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def policy_action(model, observation, device="cpu"):
    torch, _ = _require_torch()
    with torch.no_grad():
        obs = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        dist, _value = model(obs)
        return dist.mean.squeeze(0).clamp(-1.0, 1.0).cpu().tolist()


def evaluate(
    checkpoint,
    scene_path="",
    robot="robot",
    backend="null",
    project_path=None,
    config=None,
    device="cpu",
    gobot_pythonpath=None,
    env_type="rl",
    env_options=None,
    steps=1000,
    seed=1,
):
    if env_type not in ("mujoco_cartpole",):
        add_gobot_pythonpath(gobot_pythonpath)
        if project_path:
            import gobot

            gobot.app.context().set_project_path(project_path)

    cfg = config or PPOConfig()
    env = GobotGymEnv(
        scene_path=scene_path,
        robot=robot,
        backend=backend,
        env_type=env_type,
        gobot_pythonpath=gobot_pythonpath,
        project_path=project_path,
        env_options=env_options,
        action_scale=cfg.action_scale,
        action_rate_limit=cfg.action_rate_limit,
        finite_observation_limit=cfg.finite_observation_limit,
        finite_reward_limit=cfg.finite_reward_limit,
        invalid_reward=cfg.invalid_reward,
    )
    observation, info = env.reset(seed=seed)
    if not info.get("ok", True):
        raise RuntimeError(info.get("error", "failed to reset Gobot environment"))

    model = load_policy(checkpoint, len(observation), int(env.env.get_action_size()), cfg, device)
    total_reward = 0.0
    episodes = 0
    for step in range(int(steps)):
        action = policy_action(model, observation, device=device)
        observation, reward, terminated, truncated, info = env.step(action)
        if info.get("error") and not info.get("invalid_transition"):
            raise RuntimeError(info["error"])
        total_reward += float(reward)
        if terminated or truncated:
            episodes += 1
            observation, info = env.reset()
            if not info.get("ok", True):
                raise RuntimeError(info.get("error", "failed to reset Gobot environment"))
    return {"steps": int(steps), "episodes": episodes, "total_reward": total_reward}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Gobot PPO checkpoint.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gobot-pythonpath", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--robot", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    file_config = load_config_file(args.config) if args.config else {}
    env_config = _section(file_config, "env")
    ppo_values = _section(file_config, "ppo")
    config = update_dataclass(PPOConfig(), ppo_values)

    result = evaluate(
        checkpoint=Path(args.checkpoint),
        scene_path=_pick(args.scene, env_config, "scene", ""),
        robot=_pick(args.robot, env_config, "robot", "robot"),
        backend=_pick(args.backend, env_config, "backend", "null"),
        project_path=_pick(args.project, env_config, "project", None),
        config=config,
        device=_pick(args.device, env_config, "device", "cpu"),
        gobot_pythonpath=_pick(args.gobot_pythonpath, env_config, "gobot_pythonpath", None),
        env_type=env_config.get("type", "rl"),
        env_options=env_config.get("options", {}),
        steps=args.steps,
        seed=args.seed if args.seed is not None else config.seed,
    )
    print(result)


if __name__ == "__main__":
    main()
