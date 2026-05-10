"""Standalone MuJoCo CartPole environment for PPO training without gobot."""

import math
import random

import mujoco
import numpy as np

_CARTPOLE_XML = """
<mujoco model="cartpole">
  <option timestep="0.004166666666666667" integrator="RK4"/>
  <worldbody>
    <light diffuse="0.8 0.8 0.8" pos="0 0 3" dir="0 0 -1"/>
    <body name="cart" pos="0 0 0">
      <inertial pos="0 0 0" mass="1.0" diaginertia="0.02 0.02 0.02"/>
      <joint name="slider" type="slide" axis="1 0 0" limited="true" range="-2.4 2.4" damping="1.0"/>
      <geom type="box" size="0.175 0.11 0.09" rgba="0.2 0.6 0.8 1"/>
      <body name="pole" pos="0 0 0.27">
        <inertial pos="0 0 0" mass="0.3" diaginertia="0.00225 0.00225 0.000135"/>
        <joint name="hinge" type="hinge" pos="0 0 -0.15" axis="0 1 0" damping="0.05"/>
        <geom type="box" size="0.03 0.03 0.15" rgba="0.8 0.2 0.2 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slider" ctrlrange="-1 1" ctrllimited="true"/>
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
        force_limit=20.0,
        pole_angle_limit=0.4,
        cart_position_limit=2.4,
        initial_angle_range=0.05,
        target_position=1.0,
        target_tolerance=0.05,
        target_velocity_tolerance=0.15,
        settle_bonus=10.0,
        reach_bonus=20.0,
    ):
        self.max_episode_steps = int(max_episode_steps)
        self.force_limit = float(force_limit)
        self.pole_angle_limit = float(pole_angle_limit)
        self.cart_position_limit = float(cart_position_limit)
        self.initial_angle_range = float(initial_angle_range)
        self.target_position = float(target_position)
        self.target_tolerance = float(target_tolerance)
        self.target_velocity_tolerance = float(target_velocity_tolerance)
        self.settle_bonus = float(settle_bonus)
        self.reach_bonus = float(reach_bonus)
        self._rng = random.Random()

        self.model = mujoco.MjModel.from_xml_string(_CARTPOLE_XML)
        self.model.actuator_ctrlrange[0, 0] = -self.force_limit
        self.model.actuator_ctrlrange[0, 1] = self.force_limit
        self.model.actuator_gear[0, 0] = 1.0
        self.data = mujoco.MjData(self.model)
        self.episode_steps = 0
        self._previous_x = 0.0
        self._reached_target_near = False

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng.seed(int(seed))
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[1] = self._rng.uniform(-self.initial_angle_range, self.initial_angle_range)
        mujoco.mj_forward(self.model, self.data)
        self.episode_steps = 0
        self._previous_x = 0.0
        self._reached_target_near = False
        return self._observation(), {"ok": True}

    def step(self, action):
        action_value = float(action[0]) if action else 0.0
        if not math.isfinite(action_value):
            action_value = 0.0
        action_value = max(-1.0, min(1.0, action_value))

        self.data.ctrl[0] = action_value * self.force_limit
        mujoco.mj_step(self.model, self.data)
        self.episode_steps += 1

        obs = self._observation()
        x, x_dot, theta, theta_dot, target_error = obs

        terminated = (
            abs(theta) > self.pole_angle_limit
            or abs(x) > self.cart_position_limit
        )
        truncated = self.episode_steps >= self.max_episode_steps

        distance = abs(target_error)
        previous_target_error = self.target_position - self._previous_x
        previous_distance = abs(previous_target_error)
        progress = max(-0.05, min(0.05, previous_distance - distance))
        crossed_target_fast = previous_target_error * target_error < 0.0 and abs(x_dot) > self.target_velocity_tolerance
        overshoot = max(0.0, abs(x) - abs(self.target_position))

        state_cost = (
            8.0 * distance * distance
            + 0.5 * x_dot * x_dot
            + 300.0 * theta * theta
            + 8.0 * theta_dot * theta_dot
            + 20.0 * overshoot * overshoot
            + 0.02 * action_value * action_value
        )

        reward = (
            8.0
            - state_cost
            + 4.0 * progress
        )
        if (
            distance <= self.target_tolerance
            and abs(x_dot) <= self.target_velocity_tolerance
            and abs(theta) <= 0.10
        ):
            reward += self.settle_bonus
        if not self._reached_target_near and distance <= 0.1 and abs(theta) <= 0.20:
            self._reached_target_near = True
            time_left = 1.0 - min(self.episode_steps / max(self.max_episode_steps, 1), 1.0)
            reward += self.reach_bonus * time_left
        if crossed_target_fast:
            reward -= 10.0
        if terminated:
            reward = -100.0

        self._previous_x = x

        return obs, float(reward), terminated, truncated, {
            "target_error": target_error,
            "target_position_error": target_error,
            "cart_position": x,
            "cart_velocity": x_dot,
            "pole_angle": theta,
            "pole_angular_velocity": theta_dot,
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
