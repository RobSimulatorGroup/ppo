import math
import os
import random
import sys


def add_gobot_pythonpath(path=None):
    path = path or os.environ.get("GOBOT_PYTHONPATH")
    if path and path not in sys.path:
        sys.path.insert(0, path)


class GobotBox:
    def __init__(self, low, high, names=None, units=None):
        self.low = [float(value) for value in low]
        self.high = [float(value) for value in high]
        self.shape = (len(self.low),)
        self.names = list(names or [])
        self.units = list(units or [])

    def sample(self):
        values = []
        for lower, upper in zip(self.low, self.high):
            if math.isfinite(lower) and math.isfinite(upper):
                values.append(random.uniform(lower, upper))
            else:
                values.append(0.0)
        return values


def space_from_spec(spec):
    names = list(spec.get("names", []))
    lower_bounds = list(spec.get("lower_bounds", []))
    upper_bounds = list(spec.get("upper_bounds", []))
    units = list(spec.get("units", []))
    try:
        import numpy as np
        from gymnasium import spaces

        return spaces.Box(
            low=np.asarray(lower_bounds, dtype=np.float32),
            high=np.asarray(upper_bounds, dtype=np.float32),
            dtype=np.float32,
        )
    except Exception:
        return GobotBox(lower_bounds, upper_bounds, names=names, units=units)


class GobotGymEnv:
    def __init__(
        self,
        scene_path="",
        robot="robot",
        backend="null",
        env_type="rl",
        env=None,
        gobot_pythonpath=None,
        action_scale=1.0,
        action_rate_limit=None,
        finite_observation_limit=1.0e6,
        finite_reward_limit=1.0e6,
        invalid_reward=-1.0,
    ):
        add_gobot_pythonpath(gobot_pythonpath)
        import gobot

        if env is not None:
            self.env = env
        elif env_type == "cartpole":
            self.env = gobot.CartPoleEnv(scene_path, robot=robot, backend=backend)
        else:
            self.env = gobot.RLEnvironment(scene_path, robot=robot, backend=backend)
        self.action_scale = float(action_scale)
        self.action_rate_limit = None if action_rate_limit is None else float(action_rate_limit)
        self.finite_observation_limit = float(finite_observation_limit)
        self.finite_reward_limit = float(finite_reward_limit)
        self.invalid_reward = float(invalid_reward)
        self.previous_env_action = None
        self._refresh_spaces()

    def _refresh_spaces(self):
        self.observation_spec = self.env.get_observation_spec()
        self.action_spec = self.env.get_action_spec()
        self.observation_space = space_from_spec(self.observation_spec)
        self.action_space = space_from_spec(self.action_spec)

    def reset(self, seed=None, options=None):
        seed_value = 0 if seed is None else int(seed)
        observation, info = self.env.reset(seed=seed_value)
        self._refresh_spaces()
        self.previous_env_action = [0.0] * int(self.env.get_action_size())
        observation, invalid_reason = self._sanitize_observation(observation)
        if invalid_reason:
            info = dict(info)
            info["ok"] = False
            info["invalid_transition"] = True
            info["error"] = invalid_reason
        return observation, info

    def step(self, action):
        env_action = self._prepare_action(action)
        observation, reward, terminated, truncated, info = self.env.step(env_action)
        info = dict(info)
        info["env_action"] = list(env_action)

        observation, invalid_observation = self._sanitize_observation(observation)
        invalid_reward = self._invalid_reward_reason(reward)
        invalid_reason = invalid_observation or invalid_reward
        if invalid_reason:
            info["invalid_transition"] = True
            info["error"] = invalid_reason
            return observation, self.invalid_reward, True, truncated, info

        return observation, float(reward), terminated, truncated, info

    def close(self):
        pass

    def _prepare_action(self, action):
        action_size = int(self.env.get_action_size())
        values = [float(value) for value in action]
        if len(values) != action_size:
            raise ValueError(f"expected action size {action_size}, got {len(values)}")

        scaled = []
        for value in values:
            if not math.isfinite(value):
                value = 0.0
            value = max(-1.0, min(1.0, value))
            scaled.append(value * self.action_scale)

        if self.previous_env_action is not None and self.action_rate_limit is not None:
            limited = []
            for previous, target in zip(self.previous_env_action, scaled):
                delta = max(-self.action_rate_limit, min(self.action_rate_limit, target - previous))
                limited.append(previous + delta)
            scaled = limited

        self.previous_env_action = list(scaled)
        return scaled

    def _sanitize_observation(self, observation):
        values = []
        invalid_reason = ""
        for value in observation:
            number = float(value)
            if not math.isfinite(number):
                invalid_reason = "non-finite observation"
                values.append(0.0)
                continue
            if abs(number) > self.finite_observation_limit:
                invalid_reason = "huge observation"
                values.append(0.0)
                continue
            values.append(number)
        return values, invalid_reason

    def _invalid_reward_reason(self, reward):
        reward = float(reward)
        if not math.isfinite(reward):
            return "non-finite reward"
        if abs(reward) > self.finite_reward_limit:
            return "huge reward"
        return ""
