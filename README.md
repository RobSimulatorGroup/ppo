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

H2 in `/home/wqq/test_godot` with MuJoCo:

```bash
GOBOT_PYTHONPATH=/home/wqq/gobot/build_ppo/python \
uv run gobot-ppo \
  --project /home/wqq/test_godot \
  --scene res://world.jscn \
  --robot H2 \
  --backend mujoco \
  --total-steps 4096 \
  --rollout-steps 256 \
  --device cuda
```

Current H2 training is a pipeline smoke, not a stable walking policy yet.
Random initial policy actions can still make MuJoCo unstable. The next training
stability work should add action scaling, lower initial policy std, finite-state
termination, and a locomotion reward.
