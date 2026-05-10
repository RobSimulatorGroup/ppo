import sys
from pathlib import Path

import gobot


PPO_ROOT = Path("/home/wqq/gobot/ppo")
PROJECT_PATH = "/home/wqq/test_godot"
SCENE_PATH = "res://cartpole.jscn"
ROBOT = "cartpole"
SLIDER = "slider"
HINGE = "hinge"
CHECKPOINT = ""
TARGET_CART_POSITION = 1.0
STEPS_PER_PHYSICS_TICK = 1
PLAYBACK_PUSH_INTERVAL = 240
PLAYBACK_PUSH_FORCE = 0.35
PLAYBACK_PUSH_STEPS = 18

if str(PPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PPO_ROOT / "src"))

from gobot_ppo.config import PPOConfig, load_config_file, update_dataclass
from gobot_ppo.env import CartPoleSliderEnv, latest_checkpoint
from gobot_ppo.eval import load_policy, policy_action


state = {
    "env": None,
    "model": None,
    "observation": None,
    "ticks": 0,
    "playing": True,
    "checkpoint": "",
    "episode": 0,
}


def _config():
    config_path = PPO_ROOT / "configs" / "cartpole_mujoco.yaml"
    data = load_config_file(config_path)
    return data, update_dataclass(PPOConfig(), data.get("ppo", {}))


def _checkpoint_path(data):
    if CHECKPOINT:
        return Path(CHECKPOINT)
    save_dir = PPO_ROOT / data.get("ppo", {}).get("save_dir", "checkpoints/cartpole_mujoco")
    path = latest_checkpoint(save_dir)
    if path is None:
        raise RuntimeError(f"no PPO checkpoint found in {save_dir}")
    return path


def setup(seed=1):
    data, cfg = _config()
    checkpoint = _checkpoint_path(data)
    env_config = data.get("env", {})
    options = dict(env_config.get("options", {}))
    options["disturbance_force_std"] = 0.0
    options["disturbance_impulse_probability"] = 0.0
    options["max_episode_steps"] = 1000000
    options["target_cart_position"] = TARGET_CART_POSITION

    env = CartPoleSliderEnv(
        scene_path=env_config.get("scene", SCENE_PATH),
        robot=env_config.get("robot", ROBOT),
        backend=env_config.get("backend", "mujoco"),
        project_path=env_config.get("project", PROJECT_PATH),
        **options,
    )
    observation, info = env.reset(seed=seed)
    if not info.get("ok", True):
        raise RuntimeError(info.get("error", "failed to reset CartPole playback"))

    model = load_policy(
        checkpoint,
        observation_size=len(observation),
        action_size=env.get_action_size(),
        config=cfg,
        device="cpu",
    )

    state["env"] = env
    state["model"] = model
    state["observation"] = observation
    state["ticks"] = 0
    state["checkpoint"] = str(checkpoint)
    state["episode"] = 0
    state["playing"] = True
    print(f"CartPole PPO playback loaded: {checkpoint}")
    return env, model


def physics_tick(_delta_time):
    if not state["playing"]:
        return
    if state["env"] is None or state["model"] is None:
        setup()

    env = state["env"]
    model = state["model"]
    observation = state["observation"]
    for _ in range(STEPS_PER_PHYSICS_TICK):
        if (
            PLAYBACK_PUSH_INTERVAL > 0
            and state["ticks"] > 0
            and state["ticks"] % PLAYBACK_PUSH_INTERVAL == 0
        ):
            direction = -1.0 if (state["ticks"] // PLAYBACK_PUSH_INTERVAL) % 2 else 1.0
            env.push(direction * PLAYBACK_PUSH_FORCE, PLAYBACK_PUSH_STEPS)

        action = policy_action(model, observation, device="cpu")
        observation, _reward, terminated, truncated, info = env.step(action)
        state["ticks"] += 1
        if info.get("error"):
            print(info["error"])
            pause()
            return
        if terminated or truncated:
            state["episode"] += 1
            observation, reset_info = env.reset(seed=state["episode"] + 1)
            if not reset_info.get("ok", True):
                print(reset_info.get("error", "failed to reset CartPole playback"))
                pause()
                return
    state["observation"] = observation


def reset(seed=1):
    setup(seed=seed)


def pause():
    state["playing"] = False


def play():
    state["playing"] = True


def stop():
    gobot.clear_editor_physics_callback()
    state["playing"] = False


setup()
gobot.set_editor_physics_callback(physics_tick)
print("CartPole PPO playback started. Use pause(), play(), reset(), or stop().")
