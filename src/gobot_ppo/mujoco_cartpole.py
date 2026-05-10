"""Standalone MuJoCo CartPole environment for PPO training without gobot."""

import math
import random

import mujoco
import numpy as np

_CARTPOLE_XML = """
<mujoco model="cartpole">
  <option timestep="0.01" integrator="RK4"/>
  <worldbody>
    <light diffuse="0.8 0.8 0.8" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1"/>
    <body name="cart" pos="0 0 0.1">
      <joint name="slider" type="slide" axis="1 0 0" limited="true" range="-2.4 2.4" damping="1.0"/>
      <geom type="box" size="0.2 0.15 0.1" mass="1.0" rgba="0.2 0.6 0.8 1"/>
      <body name="pole" pos="0 0 0.1">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.05"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.03" mass="0.3" rgba="0.8 0.2 0.2 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slider" ctrlrange="-1 1" ctrllimited="true" gear="20"/>
  </actuator>
</mujoco>
"""


class MujocoCartPoleEnv:
    """CartPole using raw MuJoCo — no gobot dependency.

    Observation: [cart_position, cart_velocity, pole_angle, pole_angular_velocity, target_error]
    Action: [normalized_force] in [-1, 1]

    Goal: move the cart from 0 to target_position (default 1.0) while keeping
    the pole balanced.
    """

    def __init__(
        self,
        max_episode_steps=500,
        force_limit=100.0,
        pole_angle_limit=0.4,
        cart_position_limit=2.4,
        initial_angle_range=0.05,
        target_position=1.0,
    ):
        self.max_episode_steps = int(max_episode_steps)
        self.force_limit = float(force_limit)
        self.pole_angle_limit = float(pole_angle_limit)
        self.cart_position_limit = float(cart_position_limit)
        self.initial_angle_range = float(initial_angle_range)
        self.target_position = float(target_position)
        self._rng = random.Random()

        self.model = mujoco.MjModel.from_xml_string(_CARTPOLE_XML)
        self.data = mujoco.MjData(self.model)
        self.episode_steps = 0
        self._previous_x = 0.0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng.seed(int(seed))
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[1] = self._rng.uniform(
            -self.initial_angle_range, self.initial_angle_range
        )
        mujoco.mj_forward(self.model, self.data)
        self.episode_steps = 0
        self._previous_x = 0.0
        return self._observation(), {"ok": True}

    def step(self, action):
        action_value = float(action[0]) if action else 0.0
        if not math.isfinite(action_value):
            action_value = 0.0
        action_value = max(-1.0, min(1.0, action_value))

        self.data.ctrl[0] = action_value
        mujoco.mj_step(self.model, self.data)
        self.episode_steps += 1

        obs = self._observation()
        x, x_dot, theta, theta_dot, target_error = obs

        terminated = (
            abs(theta) > self.pole_angle_limit
            or abs(x) > self.cart_position_limit
        )
        truncated = self.episode_steps >= self.max_episode_steps

        self._previous_x = x

        # Simple reward: alive + position closeness + balance
        distance = abs(target_error)
        near_target = distance < 0.3
        closeness = math.exp(-3.0 * distance * distance)
        reward = (
            1.0                                    # alive bonus
            + 3.0 * closeness                      # position: peaks at target
            + 1.0 * math.cos(theta)                # balance: 1 when upright
            - 1.0 * x_dot * x_dot * closeness      # velocity penalty scaled by closeness
        )
        if near_target and abs(x_dot) < 0.15 and abs(theta) < 0.15:
            reward += 5.0  # bonus for being still at target
        if terminated:
            reward = 0.0

        return obs, float(reward), terminated, truncated, {
            "target_error": target_error,
        }

    def close(self):
        pass

    def get_observation_size(self):
        return 5

    def get_action_size(self):
        return 1

    def get_observation_spec(self):
        return {
            "names": ["cart_position", "cart_velocity", "pole_angle", "pole_angular_velocity", "target_error"],
            "lower_bounds": [-self.cart_position_limit, -10.0, -math.pi, -10.0, -2 * self.cart_position_limit],
            "upper_bounds": [self.cart_position_limit, 10.0, math.pi, 10.0, 2 * self.cart_position_limit],
            "units": ["m", "m/s", "rad", "rad/s", "m"],
        }

    def get_action_spec(self):
        return {
            "names": ["force_normalized"],
            "lower_bounds": [-1.0],
            "upper_bounds": [1.0],
            "units": ["normalized"],
        }

    def _observation(self):
        x = float(self.data.qpos[0])
        theta = float(self.data.qpos[1])
        x_dot = float(self.data.qvel[0])
        theta_dot = float(self.data.qvel[1])
        target_error = self.target_position - x
        return [x, x_dot, theta, theta_dot, target_error]
