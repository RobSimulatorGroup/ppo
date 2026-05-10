import sys
from pathlib import Path

import gobot


PPO_ROOT = Path("/home/wqq/gobot/ppo")
PROJECT_PATH = "/home/wqq/test_godot"
SCENE_PATH = "res://cartpole.jscn"
ROBOT = "cartpole"
TARGET_CART_POSITION = 1.0
FORCE_LIMIT = 20.0
STEPS_PER_PHYSICS_TICK = 1
THETA_SIGN = 1.0

if str(PPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PPO_ROOT / "src"))

from gobot_ppo.env import GobotCartPoleTargetEnv
from gobot_ppo.lqr import CartPoleLQRController


state = {
    "env": None,
    "controller": None,
    "observation": None,
    "ticks": 0,
    "playing": True,
}


def setup(seed=1):
    options = {
        "force_limit": FORCE_LIMIT,
        "target_cart_position": TARGET_CART_POSITION,
        "max_episode_steps": 1000000,
        "pole_angle_limit": 1.2,
        "cart_position_limit": 2.4,
        "initial_angle": 0.0,
        "randomize_initial_angle": False,
        "disturbance_force_std": 0.0,
        "disturbance_impulse_probability": 0.0,
    }
    env = GobotCartPoleTargetEnv(
        scene_path=SCENE_PATH,
        robot=ROBOT,
        backend="mujoco",
        project_path=PROJECT_PATH,
        **options,
    )
    observation, info = env.reset(seed=seed)
    if not info.get("ok", True):
        raise RuntimeError(info.get("error", "failed to reset CartPole LQR playback"))

    controller = CartPoleLQRController(
        target_position=TARGET_CART_POSITION,
        force_limit=FORCE_LIMIT,
        theta_sign=THETA_SIGN,
    )

    state["env"] = env
    state["controller"] = controller
    state["observation"] = observation
    state["ticks"] = 0
    state["playing"] = True
    print("CartPole LQR playback started.")
    return env, controller


def physics_tick(_delta_time):
    if not state["playing"]:
        return
    if state["env"] is None or state["controller"] is None:
        setup()

    env = state["env"]
    controller = state["controller"]
    observation = state["observation"]
    for _ in range(STEPS_PER_PHYSICS_TICK):
        action = controller.action(observation, target_position=TARGET_CART_POSITION)
        observation, _reward, terminated, truncated, info = env.step(action)
        state["ticks"] += 1
        if state["ticks"] % 240 == 0:
            x, x_dot, theta, theta_dot, target_error = observation
            print(
                "LQR t={:.2f}s x={:.3f} x_dot={:.3f} theta={:.4f} theta_dot={:.3f} target_error={:.3f} effort={:.3f}".format(
                    state["ticks"] / 240.0,
                    x,
                    x_dot,
                    theta,
                    theta_dot,
                    target_error,
                    info.get("slider_effort", controller.last_effort),
                )
            )
        if info.get("error"):
            print(info["error"])
            pause()
            return
        if terminated or truncated:
            x, x_dot, theta, theta_dot, target_error = observation
            print(
                "CartPole LQR stopped: terminated={} truncated={} x={:.3f} theta={:.4f} target_error={:.3f}".format(
                    terminated,
                    truncated,
                    x,
                    theta,
                    target_error,
                )
            )
            pause()
            return
    state["observation"] = observation


def reset(seed=1):
    setup(seed=seed)


def pause():
    state["playing"] = False


def play():
    state["playing"] = True


def set_target(position):
    global TARGET_CART_POSITION
    TARGET_CART_POSITION = float(position)
    if state["env"] is not None:
        state["env"].target_cart_position = TARGET_CART_POSITION
    if state["controller"] is not None:
        state["controller"].target_position = TARGET_CART_POSITION


def push(normalized_force=0.35, steps=18):
    if state["env"] is not None:
        state["env"].push(normalized_force, steps)


def stop():
    gobot.clear_editor_physics_callback()
    state["playing"] = False


setup()
gobot.set_editor_physics_callback(physics_tick)
print("Use pause(), play(), reset(), set_target(x), push(), or stop().")
