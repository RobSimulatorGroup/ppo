import argparse
from pathlib import Path

from .config import PPOConfig, load_config_file, update_dataclass
from .env import GobotPPOEnv
from .runner import ActorCritic, _require_torch, _transform_action


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
    action_transform = str(checkpoint_config.get("action_transform", "clamp"))
    setattr(model, "_gobot_action_transform", action_transform)
    if "obs_mean" in checkpoint and "obs_var" in checkpoint:
        torch = _require_torch()[0]
        setattr(model, "_gobot_obs_mean", torch.tensor(checkpoint["obs_mean"], dtype=torch.float32, device=device))
        setattr(model, "_gobot_obs_var", torch.tensor(checkpoint["obs_var"], dtype=torch.float32, device=device))
    return model


def policy_action(model, observation, device="cpu"):
    torch, _ = _require_torch()
    with torch.no_grad():
        obs = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        obs_mean = getattr(model, "_gobot_obs_mean", None)
        obs_var = getattr(model, "_gobot_obs_var", None)
        if obs_mean is not None and obs_var is not None:
            obs = (obs - obs_mean) / (obs_var.sqrt() + 1.0e-8)
        dist, _value = model(obs)
        action = _transform_action(torch, dist.mean, getattr(model, "_gobot_action_transform", "clamp"))
        return action.squeeze(0).cpu().tolist()


def evaluate(
    checkpoint,
    scene_path="",
    robot="robot",
    backend="null",
    project_path=None,
    config=None,
    device="cpu",
    env_type="rl",
    env_options=None,
    steps=1000,
    seed=1,
):
    cfg = config or PPOConfig()
    env = GobotPPOEnv(
        scene_path=scene_path,
        robot=robot,
        backend=backend,
        env_type=env_type,
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
    max_abs_pole_angle = 0.0
    max_abs_cart_position = 0.0
    total_abs_target_error = 0.0
    last_info = {}
    for step in range(int(steps)):
        action = policy_action(model, observation, device=device)
        observation, reward, terminated, truncated, info = env.step(action)
        if info.get("error") and not info.get("invalid_transition"):
            raise RuntimeError(info["error"])
        total_reward += float(reward)
        last_info = dict(info)
        max_abs_pole_angle = max(max_abs_pole_angle, abs(float(info.get("pole_angle", 0.0))))
        max_abs_cart_position = max(max_abs_cart_position, abs(float(info.get("cart_position", 0.0))))
        total_abs_target_error += abs(float(info.get("target_position_error", 0.0)))
        if terminated or truncated:
            episodes += 1
            observation, info = env.reset()
            if not info.get("ok", True):
                raise RuntimeError(info.get("error", "failed to reset Gobot environment"))
    return {
        "steps": int(steps),
        "episodes": episodes,
        "total_reward": total_reward,
        "mean_abs_target_error": total_abs_target_error / max(int(steps), 1),
        "final_target_error": float(last_info.get("target_position_error", 0.0)),
        "final_cart_position": float(last_info.get("cart_position", 0.0)),
        "final_cart_velocity": float(last_info.get("cart_velocity", 0.0)),
        "final_pole_angle": float(last_info.get("pole_angle", 0.0)),
        "max_abs_pole_angle": max_abs_pole_angle,
        "max_abs_cart_position": max_abs_cart_position,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Gobot PPO checkpoint.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", required=True)
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
        env_type=env_config.get("type", "rl"),
        env_options=env_config.get("options", {}),
        steps=args.steps,
        seed=args.seed if args.seed is not None else config.seed,
    )
    print(result)


if __name__ == "__main__":
    main()
