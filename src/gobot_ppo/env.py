import math
import random
from pathlib import Path


def _wrap_angle(value):
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


class VectorSpace:
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
    return VectorSpace(lower_bounds, upper_bounds, names=names, units=units)


class GobotPPOEnv:
    def __init__(
        self,
        scene_path="",
        robot="robot",
        backend="null",
        env_type="rl",
        env=None,
        project_path=None,
        env_options=None,
        action_scale=1.0,
        action_rate_limit=None,
        finite_observation_limit=1.0e6,
        finite_reward_limit=1.0e6,
        invalid_reward=-1.0,
        render_mode=None,
    ):
        self.render_mode = render_mode
        if env is not None:
            self.env = env
        elif env_type == "mujoco_cartpole":
            from .mujoco_cartpole import MujocoCartPoleEnv

            self.env = MujocoCartPoleEnv(**dict(env_options or {}))
        elif env_type == "cartpole":
            self.env = GobotCartPoleTargetEnv(
                scene_path=scene_path,
                robot=robot,
                backend=backend,
                project_path=project_path,
                **dict(env_options or {}),
            )
        else:
            raise ValueError(f"unsupported env_type: {env_type}")
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
        seed_value = None if seed is None else int(seed)
        try:
            observation, info = self.env.reset(seed=seed_value, options=options)
        except TypeError:
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

    def render(self):
        return None

    @property
    def unwrapped(self):
        return self

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


class GobotCartPoleTargetEnv:
    """CartPole task implemented above Gobot's generic scene and joint APIs.

    The policy only controls the cart slider. The pole hinge stays passive and
    is observed as ordinary joint state, which keeps the RL task entirely in
    Python and leaves Gobot's C++ side as a generic simulator/controller layer.
    """

    def __init__(
        self,
        scene_path="res://cartpole.jscn",
        robot="cartpole",
        backend="mujoco",
        project_path=None,
        slider_joint="slider",
        hinge_joint="hinge",
        force_limit=120.0,
        max_episode_steps=500,
        pole_angle_limit=0.7,
        cart_position_limit=2.4,
        initial_angle=0.12,
        initial_cart_position=0.0,
        target_cart_position=1.0,
        randomize_target_position=False,
        target_cart_position_range=(-1.0, 1.0),
        target_tolerance=0.05,
        target_near_tolerance=0.1,
        target_velocity_tolerance=0.2,
        target_overspeed_limit=0.8,
        fast_reach_bonus=20.0,
        randomize_initial_angle=False,
        disturbance_force_std=0.0,
        disturbance_impulse_probability=0.0,
        disturbance_impulse_force=0.0,
        disturbance_impulse_steps=1,
        fixed_dt=1.0 / 240.0,
    ):
        import gobot

        self.gobot = gobot
        self.context = gobot.app.context()
        self.scene_path = scene_path
        self.robot = robot
        self.backend = backend
        self.project_path = project_path
        self.slider_joint_name = slider_joint
        self.hinge_joint_name = hinge_joint
        self.force_limit = float(force_limit)
        self.max_episode_steps = int(max_episode_steps)
        self.pole_angle_limit = float(pole_angle_limit)
        self.cart_position_limit = float(cart_position_limit)
        self.initial_angle = float(initial_angle)
        self.initial_cart_position = float(initial_cart_position)
        self.default_target_cart_position = float(target_cart_position)
        self.target_cart_position = float(target_cart_position)
        self.randomize_target_position = bool(randomize_target_position)
        self.target_cart_position_range = self._parse_range(target_cart_position_range)
        self.target_tolerance = float(target_tolerance)
        self.target_near_tolerance = float(target_near_tolerance)
        self.target_velocity_tolerance = float(target_velocity_tolerance)
        self.target_overspeed_limit = float(target_overspeed_limit)
        self.fast_reach_bonus = float(fast_reach_bonus)
        self.randomize_initial_angle = bool(randomize_initial_angle)
        self.disturbance_force_std = float(disturbance_force_std)
        self.disturbance_impulse_probability = float(disturbance_impulse_probability)
        self.disturbance_impulse_force = float(disturbance_impulse_force)
        self.disturbance_impulse_steps = max(1, int(disturbance_impulse_steps))
        self._disturbance_impulse_remaining = 0
        self._disturbance_impulse_value = 0.0
        self.fixed_dt = float(fixed_dt)
        self._world_built = False
        self.episode_steps = 0
        self.previous_x = 0.0
        self.previous_theta = 0.0
        self.previous_observation = [0.0, 0.0, 0.0, 0.0, self.target_cart_position]
        self.last_error = ""
        self._reached_target_near = False
        self._rng = random.Random()
        self._root = None
        self._slider = None
        self._hinge = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng.seed(int(seed))
        options = dict(options or {})
        self.target_cart_position = self._target_position(options)

        try:
            self._load_scene_if_needed()
            self._resolve_nodes()
            if self._world_built and not self.randomize_initial_angle:
                self.context.reset_simulation()
            else:
                self._slider.joint_position = float(self.initial_cart_position)
                self._hinge.joint_position = self._initial_angle()
                self._slider.effort_limit = self.force_limit
                self._slider.velocity_limit = max(float(getattr(self._slider, "velocity_limit", 0.0)), 20.0)
                self.context.build_world(self._backend_type())
                self._world_built = True
            self.context.set_joint_passive(self.robot, self.hinge_joint_name)
            self.context.set_joint_passive(self.robot, self.slider_joint_name)
        except Exception as error:
            self.last_error = str(error)
            return [0.0, 0.0, 0.0, 0.0, self.target_cart_position], {"ok": False, "error": self.last_error}

        self.episode_steps = 0
        self.previous_x = float(self._slider.joint_position)
        self.previous_theta = _wrap_angle(float(self._hinge.joint_position))
        self.previous_observation = self._observation(self.previous_x, 0.0, self.previous_theta, 0.0)
        self._disturbance_impulse_remaining = 0
        self._disturbance_impulse_value = 0.0
        self._reached_target_near = False
        self.last_error = ""
        return list(self.previous_observation), {
            "ok": True,
            "seed": seed,
            "target_cart_position": self.target_cart_position,
            "frame_count": int(self.context.frame_count),
            "simulation_time": float(self.context.simulation_time),
        }

    def step(self, action):
        try:
            self._resolve_nodes()
            action_value = float(action[0]) if action else 0.0
            if not math.isfinite(action_value):
                action_value = 0.0
            action_value = max(-1.0, min(1.0, action_value))
            policy_effort = action_value * self.force_limit
            disturbance_effort = self._sample_disturbance_effort()
            effort = max(-self.force_limit, min(self.force_limit, policy_effort + disturbance_effort))
            self.context.set_joint_passive(self.robot, self.hinge_joint_name)
            self.context.set_joint_effort_target(self.robot, self.slider_joint_name, effort)
            self.context.step_once()
            self._resolve_nodes()
        except Exception as error:
            self.last_error = str(error)
            return list(self.previous_observation), 0.0, True, False, {"error": self.last_error}

        x = float(self._slider.joint_position)
        theta = _wrap_angle(float(self._hinge.joint_position))
        x_dot = (x - self.previous_x) / self.fixed_dt
        theta_dot = _wrap_angle(theta - self.previous_theta) / self.fixed_dt
        observation = self._observation(x, x_dot, theta, theta_dot)
        previous_target_error = self.target_cart_position - self.previous_x

        self.previous_x = x
        self.previous_theta = theta
        self.previous_observation = list(observation)
        self.episode_steps += 1

        terminated = (
            abs(x) > self.cart_position_limit
            or abs(theta) > self.pole_angle_limit
        )
        truncated = self.episode_steps >= self.max_episode_steps
        oversped_target = (
            abs(observation[4]) <= self.target_near_tolerance
            and abs(x_dot) > self.target_overspeed_limit
        )
        reward = self._reward(observation, action_value, terminated, previous_target_error, oversped_target)
        self.last_error = ""
        return observation, reward, terminated, truncated, {
            "frame_count": int(self.context.frame_count),
            "simulation_time": float(self.context.simulation_time),
            "target_cart_position": self.target_cart_position,
            "target_position_error": self.target_cart_position - x,
            "cart_position": x,
            "cart_velocity": x_dot,
            "pole_angle": theta,
            "pole_angular_velocity": theta_dot,
            "slider_effort": effort,
            "policy_effort": policy_effort,
            "disturbance_effort": disturbance_effort,
            "oversped_target": oversped_target,
        }

    def close(self):
        pass

    def push(self, normalized_force=0.35, steps=None):
        value = max(-1.0, min(1.0, float(normalized_force))) * self.force_limit
        self._disturbance_impulse_value = value
        self._disturbance_impulse_remaining = max(1, int(steps or self.disturbance_impulse_steps))

    def get_action_size(self):
        return 1

    def get_observation_size(self):
        return 5

    def get_action_spec(self):
        return {
            "version": "cartpole_slider_v1",
            "names": ["slider_effort_normalized"],
            "lower_bounds": [-1.0],
            "upper_bounds": [1.0],
            "units": ["normalized"],
        }

    def get_observation_spec(self):
        return {
            "version": "cartpole_slider_v1",
            "names": [
                "cart_position",
                "cart_velocity",
                "pole_angle",
                "pole_angular_velocity",
                "target_position_error",
            ],
            "lower_bounds": [
                -self.cart_position_limit,
                -math.inf,
                -math.pi,
                -math.inf,
                -2.0 * self.cart_position_limit,
            ],
            "upper_bounds": [
                self.cart_position_limit,
                math.inf,
                math.pi,
                math.inf,
                2.0 * self.cart_position_limit,
            ],
            "units": ["m", "m/s", "rad", "rad/s", "m"],
        }

    def get_last_error(self):
        return self.last_error

    def _load_scene_if_needed(self):
        if self.project_path:
            self.context.set_project_path(str(self.project_path))
        if not self.scene_path:
            raise RuntimeError("GobotCartPoleTargetEnv requires a Gobot scene path.")
        if not self.context.has_scene or self.context.scene_path != self.scene_path:
            self.context.load_scene(self.scene_path)
        self._root = self.context.root
        if self._root is None:
            raise RuntimeError(f"failed to load Gobot scene '{self.scene_path}'")

    def _resolve_nodes(self):
        root = self.context.root
        if root is None:
            raise RuntimeError("active Gobot context has no scene root")
        robot_root = root if root.name == self.robot else root.find(self.robot)
        if robot_root is None:
            raise RuntimeError(f"scene has no robot node '{self.robot}'")
        slider = self._find_joint(robot_root, self.slider_joint_name)
        hinge = self._find_joint(robot_root, self.hinge_joint_name)
        if slider is None:
            raise RuntimeError(f"robot '{self.robot}' has no slider joint '{self.slider_joint_name}'")
        if hinge is None:
            raise RuntimeError(f"robot '{self.robot}' has no hinge joint '{self.hinge_joint_name}'")
        self._root = robot_root
        self._slider = slider
        self._hinge = hinge

    def _find_joint(self, root, name):
        direct_paths = {
            self.slider_joint_name: ["rail/slider", "slider"],
            self.hinge_joint_name: ["rail/slider/cart/hinge", "cart/hinge", "hinge"],
        }.get(name, [name])
        for path in direct_paths:
            node = root.find(path)
            if node is not None:
                return node
        return self._find_node_by_name(root, name)

    def _find_node_by_name(self, node, name):
        if node.name == name:
            return node
        for child in node.children:
            found = self._find_node_by_name(child, name)
            if found is not None:
                return found
        return None

    def _backend_type(self):
        backend = str(self.backend).lower()
        if backend == "mujoco":
            return self.gobot.PhysicsBackendType.MuJoCoCpu
        if backend == "null":
            return self.gobot.PhysicsBackendType.Null
        raise ValueError(f"unsupported Gobot backend for GobotCartPoleTargetEnv: {self.backend}")

    def _initial_angle(self):
        if self.randomize_initial_angle:
            return self._rng.uniform(-self.initial_angle, self.initial_angle)
        return self.initial_angle

    def _target_position(self, options):
        if "target_cart_position" in options:
            return float(options["target_cart_position"])
        if self.randomize_target_position:
            return self._rng.uniform(*self.target_cart_position_range)
        return self.default_target_cart_position

    def _parse_range(self, value):
        values = list(value)
        if len(values) != 2:
            raise ValueError("target_cart_position_range must contain [min, max]")
        lower = float(values[0])
        upper = float(values[1])
        if lower > upper:
            lower, upper = upper, lower
        return lower, upper

    def _observation(self, x, x_dot, theta, theta_dot):
        return [x, x_dot, theta, theta_dot, self.target_cart_position - x]

    def _sample_disturbance_effort(self):
        effort = 0.0
        if self.disturbance_force_std > 0.0:
            effort += self._rng.gauss(0.0, self.disturbance_force_std)

        if self._disturbance_impulse_remaining > 0:
            self._disturbance_impulse_remaining -= 1
            effort += self._disturbance_impulse_value
        elif (
            self.disturbance_impulse_probability > 0.0
            and self.disturbance_impulse_force > 0.0
            and self._rng.random() < self.disturbance_impulse_probability
        ):
            self._disturbance_impulse_remaining = self.disturbance_impulse_steps - 1
            self._disturbance_impulse_value = self._rng.choice((-1.0, 1.0)) * self.disturbance_impulse_force
            effort += self._disturbance_impulse_value

        return effort

    def _reward(self, observation, action, terminated, previous_target_error, oversped_target):
        x, x_dot, theta, theta_dot, target_error = observation
        distance = abs(target_error)
        previous_distance = abs(previous_target_error)
        progress = max(-0.05, min(0.05, previous_distance - distance))
        crossed_target_fast = previous_target_error * target_error < 0.0 and abs(x_dot) > self.target_velocity_tolerance
        overshoot = max(0.0, abs(x) - abs(self.target_cart_position))

        state_cost = (
            8.0 * distance * distance
            + 0.5 * x_dot * x_dot
            + 300.0 * theta * theta
            + 8.0 * theta_dot * theta_dot
            + 20.0 * overshoot * overshoot
            + 0.02 * action * action
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
            reward += 8.0
        if (
            not self._reached_target_near
            and distance <= self.target_near_tolerance
            and abs(theta) <= 0.20
        ):
            self._reached_target_near = True
            time_left = 1.0 - min(self.episode_steps / max(self.max_episode_steps, 1), 1.0)
            reward += self.fast_reach_bonus * time_left
        if oversped_target:
            reward -= 10.0
        if crossed_target_fast:
            reward -= 10.0
        if terminated:
            reward = -100.0
        return float(reward)


def latest_checkpoint(directory):
    paths = sorted(Path(directory).glob("ppo_steps_*.pt"))
    if not paths:
        return None
    return max(paths, key=lambda path: int(path.stem.split("_")[-1]))
