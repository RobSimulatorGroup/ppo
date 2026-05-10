import csv
from dataclasses import asdict
from pathlib import Path

from .config import PPOConfig
from .env import GobotPPOEnv


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError(
            "gobot-ppo requires PyTorch. Install dependencies with `uv sync`."
        ) from error
    return torch, nn


def _squash_action(torch, raw_action):
    return torch.tanh(raw_action)


def _squashed_log_prob(torch, dist, raw_action):
    squashed_action = _squash_action(torch, raw_action)
    log_prob = dist.log_prob(raw_action).sum(-1)
    log_det_jacobian = torch.log(1.0 - squashed_action.pow(2) + 1.0e-6).sum(-1)
    return log_prob - log_det_jacobian


def _clip_action(raw_action):
    return raw_action.clamp(-1.0, 1.0)


def _transform_action(torch, raw_action, mode):
    if str(mode).lower() == "tanh":
        return _squash_action(torch, raw_action)
    return _clip_action(raw_action)


def _action_log_prob(torch, dist, raw_action, mode):
    if str(mode).lower() == "tanh":
        return _squashed_log_prob(torch, dist, raw_action)
    return dist.log_prob(raw_action).sum(-1)


def _init_obs_stats(owner, observation_size):
    torch = owner.torch
    owner.obs_mean = torch.zeros(int(observation_size), dtype=torch.float32, device=owner.device)
    owner.obs_var = torch.ones(int(observation_size), dtype=torch.float32, device=owner.device)
    owner.obs_count = torch.tensor(1.0e-4, dtype=torch.float32, device=owner.device)


def _update_obs_stats(owner, observations):
    if not owner.config.normalize_observations:
        return
    torch = owner.torch
    batch = observations.detach()
    if batch.ndim == 1:
        batch = batch.unsqueeze(0)
    batch_count = torch.tensor(float(batch.shape[0]), dtype=torch.float32, device=owner.device)
    batch_mean = batch.mean(dim=0)
    batch_var = batch.var(dim=0, unbiased=False)
    delta = batch_mean - owner.obs_mean
    total_count = owner.obs_count + batch_count
    new_mean = owner.obs_mean + delta * batch_count / total_count
    mean_a = owner.obs_var * owner.obs_count
    mean_b = batch_var * batch_count
    correction = delta.pow(2) * owner.obs_count * batch_count / total_count
    new_var = (mean_a + mean_b + correction) / total_count
    owner.obs_mean = new_mean
    owner.obs_var = new_var.clamp_min(1.0e-6)
    owner.obs_count = total_count


def _normalize_obs(owner, observations):
    if not owner.config.normalize_observations:
        return observations
    return (observations - owner.obs_mean) / (owner.obs_var.sqrt() + 1.0e-8)


def _obs_stats_checkpoint(owner):
    if not owner.config.normalize_observations:
        return {}
    return {
        "obs_mean": owner.obs_mean.detach().cpu().tolist(),
        "obs_var": owner.obs_var.detach().cpu().tolist(),
        "obs_count": float(owner.obs_count.detach().cpu()),
    }


def _load_obs_stats(owner, checkpoint):
    if "obs_mean" not in checkpoint or "obs_var" not in checkpoint:
        return
    torch = owner.torch
    owner.obs_mean = torch.tensor(checkpoint["obs_mean"], dtype=torch.float32, device=owner.device)
    owner.obs_var = torch.tensor(checkpoint["obs_var"], dtype=torch.float32, device=owner.device)
    owner.obs_count = torch.tensor(float(checkpoint.get("obs_count", 1.0)), dtype=torch.float32, device=owner.device)


class ActorCritic:
    def __init__(
        self,
        observation_size,
        action_size,
        hidden_size=128,
        initial_log_std=-1.0,
        min_log_std=-5.0,
        max_log_std=2.0,
    ):
        torch, nn = _require_torch()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.actor = nn.Sequential(
                    nn.Linear(observation_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, action_size),
                )
                self.critic = nn.Sequential(
                    nn.Linear(observation_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, 1),
                )
                self.log_std = nn.Parameter(torch.full((action_size,), float(initial_log_std)))
                self.min_log_std = float(min_log_std)
                self.max_log_std = float(max_log_std)

            def forward(self, observations):
                mean = self.actor(observations)
                log_std = self.log_std.clamp(self.min_log_std, self.max_log_std)
                std = log_std.exp().expand_as(mean)
                dist = torch.distributions.Normal(mean, std)
                value = self.critic(observations).squeeze(-1)
                return dist, value

        self.model = Model()


class PPORunner:
    def __init__(self, env, config=None, device="cpu"):
        self.config = config or PPOConfig()
        self.env = env
        self.device = device
        self.torch, _ = _require_torch()
        self.torch.manual_seed(self.config.seed)

        observation_size = int(env.env.get_observation_size())
        action_size = int(env.env.get_action_size())
        if observation_size <= 0 or action_size <= 0:
            observation, info = env.reset(seed=self.config.seed)
            if not info.get("ok", True):
                raise RuntimeError(info.get("error", "failed to reset Gobot environment"))
            observation_size = len(observation)
            action_size = int(env.env.get_action_size())
        if observation_size <= 0 or action_size <= 0:
            raise RuntimeError("Gobot PPO requires non-empty observation and action spaces.")

        self.agent = ActorCritic(
            observation_size,
            action_size,
            hidden_size=self.config.hidden_size,
            initial_log_std=self.config.initial_log_std,
            min_log_std=self.config.min_log_std,
            max_log_std=self.config.max_log_std,
        ).model.to(device)
        self.optimizer = self.torch.optim.Adam(self.agent.parameters(), lr=self.config.learning_rate)
        self.start_steps = 0
        self.start_episodes = 0
        self._last_save_steps = 0
        _init_obs_stats(self, observation_size)
        self._init_log()
        if self.config.resume:
            self._load_checkpoint(self.config.resume)

    def train(self):
        torch = self.torch
        cfg = self.config
        observation, info = self.env.reset(seed=cfg.seed)
        if not info.get("ok", True):
            raise RuntimeError(info.get("error", "failed to reset Gobot environment"))

        obs = torch.tensor(observation, dtype=torch.float32, device=self.device)
        _update_obs_stats(self, obs)
        completed_steps = self.start_steps
        episode_return = 0.0
        episode_count = self.start_episodes
        last_loss = 0.0

        while completed_steps < cfg.total_steps:
            rollout = self._collect_rollout(obs)
            obs = rollout["next_obs"]
            episode_return += float(rollout["reward_sum"])
            episode_count += int(rollout["episode_count"])
            completed_steps += len(rollout["rewards"])
            last_loss = self._update(rollout)
            if completed_steps % max(cfg.rollout_steps, 1) == 0:
                mean_reward = episode_return / max(episode_count, 1)
                print(f"steps={completed_steps} episodes={episode_count} mean_reward={mean_reward:.6f} loss={last_loss:.6f}")
                self._write_log(completed_steps, episode_count, mean_reward, last_loss, rollout["last_info"])
                self._save_checkpoint_if_needed(completed_steps, episode_count, last_loss)

        return {"steps": completed_steps, "episodes": episode_count, "last_loss": last_loss}

    def _collect_rollout(self, obs):
        torch = self.torch
        cfg = self.config
        observations = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        reward_sum = 0.0
        episode_count = 0
        last_info = {}

        for _ in range(cfg.rollout_steps):
            with torch.no_grad():
                normalized_obs = _normalize_obs(self, obs)
                dist, value = self.agent(normalized_obs.unsqueeze(0))
                raw_action = dist.sample()
                action = _transform_action(torch, raw_action, cfg.action_transform)
                log_prob = _action_log_prob(torch, dist, raw_action, cfg.action_transform)

            next_observation, reward, terminated, truncated, info = self.env.step(
                action.squeeze(0).cpu().tolist()
            )
            if info.get("error") and not info.get("invalid_transition"):
                raise RuntimeError(info["error"])
            last_info = dict(info)

            done = bool(terminated or truncated)
            observations.append(normalized_obs)
            actions.append(raw_action.squeeze(0))
            log_probs.append(log_prob.squeeze(0))
            rewards.append(float(reward) * cfg.reward_scale)
            dones.append(done)
            values.append(value.squeeze(0))
            reward_sum += float(reward)

            if done:
                next_observation, reset_info = self.env.reset()
                if not reset_info.get("ok", True):
                    raise RuntimeError(reset_info.get("error", "failed to reset Gobot environment"))
                episode_count += 1
            obs = torch.tensor(next_observation, dtype=torch.float32, device=self.device)
            _update_obs_stats(self, obs)

        with torch.no_grad():
            _, next_value = self.agent(_normalize_obs(self, obs).unsqueeze(0))

        return {
            "observations": torch.stack(observations),
            "actions": torch.stack(actions),
            "log_probs": torch.stack(log_probs),
            "rewards": torch.tensor(rewards, dtype=torch.float32, device=self.device),
            "dones": torch.tensor(dones, dtype=torch.float32, device=self.device),
            "values": torch.stack(values),
            "next_value": next_value.squeeze(0),
            "next_obs": obs,
            "reward_sum": reward_sum,
            "episode_count": episode_count,
            "last_info": last_info,
        }

    def _update(self, rollout):
        torch = self.torch
        cfg = self.config
        rewards = rollout["rewards"]
        dones = rollout["dones"]
        values = rollout["values"].detach()
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        next_value = rollout["next_value"].detach()

        for step in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[step]
            next_values = next_value if step == len(rewards) - 1 else values[step + 1]
            delta = rewards[step] + cfg.gamma * next_values * next_non_terminal - values[step]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_gae
            advantages[step] = last_gae
        returns = advantages + values

        observations = rollout["observations"]
        actions = rollout["actions"]
        old_log_probs = rollout["log_probs"].detach()
        batch_size = len(rewards)
        minibatch_size = min(cfg.minibatch_size, batch_size)
        indices = torch.arange(batch_size, device=self.device)
        last_loss = 0.0

        for _ in range(cfg.update_epochs):
            shuffled = indices[torch.randperm(batch_size, device=self.device)]
            for start in range(0, batch_size, minibatch_size):
                minibatch = shuffled[start:start + minibatch_size]
                dist, new_value = self.agent(observations[minibatch])
                new_log_prob = _action_log_prob(torch, dist, actions[minibatch], cfg.action_transform)
                entropy = dist.entropy().sum(-1).mean()
                log_ratio = new_log_prob - old_log_probs[minibatch]
                ratio = log_ratio.exp()
                mb_advantages = advantages[minibatch]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std(unbiased=False) + 1e-8)
                policy_loss = -torch.min(
                    mb_advantages * ratio,
                    mb_advantages * ratio.clamp(1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef),
                ).mean()
                value_loss = 0.5 * (returns[minibatch] - new_value).pow(2).mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                last_loss = float(loss.detach().cpu())

        return last_loss

    def _init_log(self):
        self.log_path = Path(self.config.log_path) if self.config.log_path else None
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as log_file:
                writer = csv.writer(log_file)
                writer.writerow([
                    "steps",
                    "episodes",
                    "mean_reward",
                    "loss",
                    "target_position_error",
                    "cart_position",
                    "cart_velocity",
                    "pole_angle",
                    "pole_angular_velocity",
                ])

    def _write_log(self, steps, episodes, mean_reward, loss, info=None):
        if self.log_path is None:
            return
        info = dict(info or {})
        with self.log_path.open("a", newline="", encoding="utf-8") as log_file:
            writer = csv.writer(log_file)
            writer.writerow([
                steps,
                episodes,
                f"{mean_reward:.8f}",
                f"{loss:.8f}",
                f"{float(info.get('target_position_error', 0.0)):.8f}",
                f"{float(info.get('cart_position', 0.0)):.8f}",
                f"{float(info.get('cart_velocity', 0.0)):.8f}",
                f"{float(info.get('pole_angle', 0.0)):.8f}",
                f"{float(info.get('pole_angular_velocity', 0.0)):.8f}",
            ])

    def _checkpoint_path(self, steps):
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir / f"ppo_steps_{steps}.pt"

    def _save_checkpoint_if_needed(self, steps, episodes, loss):
        if self.config.save_every <= 0:
            return
        if steps - self._last_save_steps < self.config.save_every and steps < self.config.total_steps:
            return
        path = self._checkpoint_path(steps)
        self.torch.save(
            {
                "steps": steps,
                "episodes": episodes,
                "loss": loss,
                "config": asdict(self.config),
                "model": self.agent.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                **_obs_stats_checkpoint(self),
            },
            path,
        )
        self._last_save_steps = steps
        print(f"saved_checkpoint={path}")

    def _load_checkpoint(self, path):
        try:
            checkpoint = self.torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = self.torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        _load_obs_stats(self, checkpoint)
        self.start_steps = int(checkpoint.get("steps", 0))
        self.start_episodes = int(checkpoint.get("episodes", 0))
        self._last_save_steps = self.start_steps
        print(f"loaded_checkpoint={path} steps={self.start_steps}")


def train(
    scene_path="",
    robot="robot",
    backend="null",
    project_path=None,
    config=None,
    device="cpu",
    env_type="rl",
    env_options=None,
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
    runner = PPORunner(env, config=cfg, device=device)
    return runner.train()


class VectorizedPPORunner:
    def __init__(self, envs, config=None, device="cpu"):
        self.config = config or PPOConfig()
        self.envs = list(envs)
        if not self.envs:
            raise RuntimeError("VectorizedPPORunner requires at least one environment.")
        self.num_envs = len(self.envs)
        self.device = device
        self.torch, _ = _require_torch()
        self.torch.manual_seed(self.config.seed)

        observation, info = self.envs[0].reset(seed=self.config.seed)
        if not info.get("ok", True):
            raise RuntimeError(info.get("error", "failed to reset environment"))
        observation_size = len(observation)
        action_size = int(self.envs[0].env.get_action_size())
        for env in self.envs[1:]:
            env.reset()

        self.agent = ActorCritic(
            observation_size,
            action_size,
            hidden_size=self.config.hidden_size,
            initial_log_std=self.config.initial_log_std,
            min_log_std=self.config.min_log_std,
            max_log_std=self.config.max_log_std,
        ).model.to(device)
        self.optimizer = self.torch.optim.Adam(self.agent.parameters(), lr=self.config.learning_rate)
        self.start_steps = 0
        self.start_episodes = 0
        self._last_save_steps = 0
        self._next_reset_seed = int(self.config.seed) + 1
        _init_obs_stats(self, observation_size)
        self._init_log()
        if self.config.resume:
            self._load_checkpoint(self.config.resume)

    def train(self):
        torch = self.torch
        cfg = self.config
        observations = []
        for index, env in enumerate(self.envs):
            observation, info = env.reset(seed=cfg.seed + index)
            if not info.get("ok", True):
                raise RuntimeError(info.get("error", "failed to reset environment"))
            observations.append(observation)
        obs = torch.tensor(observations, dtype=torch.float32, device=self.device)
        _update_obs_stats(self, obs)

        completed_steps = self.start_steps
        episode_returns = [0.0] * self.num_envs
        completed_episode_returns = []
        episode_count = self.start_episodes
        last_loss = 0.0

        while completed_steps < cfg.total_steps:
            rollout = self._collect_rollout(obs, episode_returns, completed_episode_returns)
            obs = rollout["next_obs"]
            episode_count += int(rollout["episode_count"])
            completed_steps += int(rollout["env_steps"])
            last_loss = self._update(rollout)
            mean_reward = (
                sum(completed_episode_returns[-100:]) / min(len(completed_episode_returns), 100)
                if completed_episode_returns
                else sum(episode_returns) / max(self.num_envs, 1)
            )
            print(f"steps={completed_steps} episodes={episode_count} mean_reward={mean_reward:.6f} loss={last_loss:.6f}")
            self._write_log(completed_steps, episode_count, mean_reward, last_loss, rollout["last_info"])
            self._save_checkpoint_if_needed(completed_steps, episode_count, last_loss)

        return {"steps": completed_steps, "episodes": episode_count, "last_loss": last_loss}

    def _collect_rollout(self, obs, episode_returns, completed_episode_returns):
        torch = self.torch
        cfg = self.config
        observations = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        episode_count = 0
        last_info = {}

        for _ in range(cfg.rollout_steps):
            with torch.no_grad():
                normalized_obs = _normalize_obs(self, obs)
                dist, value = self.agent(normalized_obs)
                raw_action = dist.sample()
                action = _transform_action(torch, raw_action, cfg.action_transform)
                log_prob = _action_log_prob(torch, dist, raw_action, cfg.action_transform)

            action_rows = action.detach().cpu().tolist()
            next_observations = []
            reward_values = []
            done_values = []
            for env_index, env in enumerate(self.envs):
                next_observation, reward, terminated, truncated, info = env.step(action_rows[env_index])
                if info.get("error") and not info.get("invalid_transition"):
                    raise RuntimeError(info["error"])
                done = bool(terminated or truncated)
                episode_returns[env_index] += float(reward)
                last_info = dict(info)
                if done:
                    completed_episode_returns.append(float(episode_returns[env_index]))
                    episode_returns[env_index] = 0.0
                    episode_count += 1
                    next_observation, reset_info = env.reset(seed=self._next_reset_seed)
                    self._next_reset_seed += 1
                    if not reset_info.get("ok", True):
                        raise RuntimeError(reset_info.get("error", "failed to reset environment"))
                next_observations.append(next_observation)
                reward_values.append(float(reward) * cfg.reward_scale)
                done_values.append(done)

            observations.append(normalized_obs)
            actions.append(raw_action)
            log_probs.append(log_prob)
            rewards.append(torch.tensor(reward_values, dtype=torch.float32, device=self.device))
            dones.append(torch.tensor(done_values, dtype=torch.float32, device=self.device))
            values.append(value)
            obs = torch.tensor(next_observations, dtype=torch.float32, device=self.device)
            _update_obs_stats(self, obs)

        with torch.no_grad():
            _, next_value = self.agent(_normalize_obs(self, obs))

        return {
            "observations": torch.stack(observations),
            "actions": torch.stack(actions),
            "log_probs": torch.stack(log_probs),
            "rewards": torch.stack(rewards),
            "dones": torch.stack(dones),
            "values": torch.stack(values),
            "next_value": next_value,
            "next_obs": obs,
            "episode_count": episode_count,
            "last_info": last_info,
            "env_steps": cfg.rollout_steps * self.num_envs,
        }

    def _update(self, rollout):
        torch = self.torch
        cfg = self.config
        rewards = rollout["rewards"]
        dones = rollout["dones"]
        values = rollout["values"].detach()
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        next_value = rollout["next_value"].detach()

        for step in reversed(range(rewards.shape[0])):
            next_non_terminal = 1.0 - dones[step]
            next_values = next_value if step == rewards.shape[0] - 1 else values[step + 1]
            delta = rewards[step] + cfg.gamma * next_values * next_non_terminal - values[step]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_gae
            advantages[step] = last_gae
        returns = advantages + values

        observations = rollout["observations"].reshape(-1, rollout["observations"].shape[-1])
        actions = rollout["actions"].reshape(-1, rollout["actions"].shape[-1])
        old_log_probs = rollout["log_probs"].reshape(-1).detach()
        returns = returns.reshape(-1)
        advantages = advantages.reshape(-1)
        batch_size = observations.shape[0]
        minibatch_size = min(cfg.minibatch_size, batch_size)
        indices = torch.arange(batch_size, device=self.device)
        last_loss = 0.0

        for _ in range(cfg.update_epochs):
            shuffled = indices[torch.randperm(batch_size, device=self.device)]
            for start in range(0, batch_size, minibatch_size):
                minibatch = shuffled[start:start + minibatch_size]
                dist, new_value = self.agent(observations[minibatch])
                new_log_prob = _action_log_prob(torch, dist, actions[minibatch], cfg.action_transform)
                entropy = dist.entropy().sum(-1).mean()
                log_ratio = new_log_prob - old_log_probs[minibatch]
                ratio = log_ratio.exp()
                mb_advantages = advantages[minibatch]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std(unbiased=False) + 1e-8)
                policy_loss = -torch.min(
                    mb_advantages * ratio,
                    mb_advantages * ratio.clamp(1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef),
                ).mean()
                value_loss = 0.5 * (returns[minibatch] - new_value).pow(2).mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                last_loss = float(loss.detach().cpu())

        return last_loss

    def _init_log(self):
        self.log_path = Path(self.config.log_path) if self.config.log_path else None
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as log_file:
                writer = csv.writer(log_file)
                writer.writerow([
                    "steps",
                    "episodes",
                    "mean_reward",
                    "loss",
                    "target_position_error",
                    "cart_position",
                    "cart_velocity",
                    "pole_angle",
                    "pole_angular_velocity",
                ])

    def _write_log(self, steps, episodes, mean_reward, loss, info=None):
        if self.log_path is None:
            return
        info = dict(info or {})
        with self.log_path.open("a", newline="", encoding="utf-8") as log_file:
            writer = csv.writer(log_file)
            writer.writerow([
                steps,
                episodes,
                f"{mean_reward:.8f}",
                f"{loss:.8f}",
                f"{float(info.get('target_position_error', 0.0)):.8f}",
                f"{float(info.get('cart_position', 0.0)):.8f}",
                f"{float(info.get('cart_velocity', 0.0)):.8f}",
                f"{float(info.get('pole_angle', 0.0)):.8f}",
                f"{float(info.get('pole_angular_velocity', 0.0)):.8f}",
            ])

    def _checkpoint_path(self, steps):
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir / f"ppo_steps_{steps}.pt"

    def _save_checkpoint_if_needed(self, steps, episodes, loss):
        if self.config.save_every <= 0:
            return
        if steps - self._last_save_steps < self.config.save_every and steps < self.config.total_steps:
            return
        path = self._checkpoint_path(steps)
        self.torch.save(
            {
                "steps": steps,
                "episodes": episodes,
                "loss": loss,
                "config": asdict(self.config),
                "model": self.agent.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                **_obs_stats_checkpoint(self),
            },
            path,
        )
        self._last_save_steps = steps
        print(f"saved_checkpoint={path}")

    def _load_checkpoint(self, path):
        try:
            checkpoint = self.torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = self.torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        _load_obs_stats(self, checkpoint)
        self.start_steps = int(checkpoint.get("steps", 0))
        self.start_episodes = int(checkpoint.get("episodes", 0))
        self._last_save_steps = self.start_steps
        print(f"loaded_checkpoint={path} steps={self.start_steps}")


def train_vectorized_mujoco(env_options=None, config=None, device="cpu"):
    from .mujoco_cartpole import MujocoCartPoleEnv

    cfg = config or PPOConfig()
    envs = [
        GobotPPOEnv(
            env=MujocoCartPoleEnv(**dict(env_options or {})),
            action_scale=cfg.action_scale,
            action_rate_limit=cfg.action_rate_limit,
            finite_observation_limit=cfg.finite_observation_limit,
            finite_reward_limit=cfg.finite_reward_limit,
            invalid_reward=cfg.invalid_reward,
        )
        for _ in range(max(1, int(cfg.num_envs)))
    ]
    runner = VectorizedPPORunner(envs, config=cfg, device=device)
    return runner.train()
