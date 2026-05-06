# Gobot PPO

PPO training code for Gobot environments.

This repository owns the RL algorithm, training CLI, and experiment-facing
configuration. The Gobot repository owns the simulator, scene loading, physics
backends, and the `gobot` Python extension.

## Relationship With Gobot

Build Gobot's Python extension for the Python version used by this project,
then point this repo at that build output:

```bash
export GOBOT_PYTHONPATH=/home/wqq/gobot/build_ppo/python
```

The trainer adds `GOBOT_PYTHONPATH` to `sys.path` before importing `gobot`.
You can also use normal `PYTHONPATH`:

```bash
PYTHONPATH=/home/wqq/gobot/build_ppo/python uv run gobot-ppo ...
```

## Setup With uv

```bash
cd /home/wqq/ppo
uv sync
```

For CUDA, make sure the selected `torch` wheel can see the driver:

```bash
uv run python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

## Smoke Tests

Null backend:

```bash
GOBOT_PYTHONPATH=/home/wqq/gobot/build_ppo/python \
uv run gobot-ppo --total-steps 8 --rollout-steps 4 --device cpu
```

a2 in `/home/wqq/test_godot` with MuJoCo:

```bash
uv run gobot-ppo --config configs/a2_mujoco.yaml
```

Current a2 training is a pipeline smoke, not a stable walking policy yet.
The trainer now applies conservative action scaling, action-rate limiting,
finite observation/reward checks, CSV logging, and periodic checkpointing. These
controls reduce MuJoCo blow-ups while the lower-level Gobot locomotion reset and
reward model continue to mature.

Override any config value from the command line:

```bash
uv run gobot-ppo --config configs/a2_mujoco.yaml \
  --total-steps 8192 \
  --action-scale 0.15 \
  --action-rate-limit 0.02
```

Resume from a checkpoint:

```bash
uv run gobot-ppo --config configs/a2_mujoco.yaml \
  --resume checkpoints/a2_mujoco/ppo_steps_4096.pt \
  --total-steps 8192
```
