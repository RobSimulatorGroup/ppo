import argparse

from .config import PPOConfig
from .runner import train


def main():
    parser = argparse.ArgumentParser(description="Train a single Gobot RL environment with PPO.")
    parser.add_argument("--gobot-pythonpath", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--scene", default="")
    parser.add_argument("--robot", default="robot")
    parser.add_argument("--backend", default="null")
    parser.add_argument("--total-steps", type=int, default=PPOConfig.total_steps)
    parser.add_argument("--rollout-steps", type=int, default=PPOConfig.rollout_steps)
    parser.add_argument("--update-epochs", type=int, default=PPOConfig.update_epochs)
    parser.add_argument("--minibatch-size", type=int, default=PPOConfig.minibatch_size)
    parser.add_argument("--learning-rate", type=float, default=PPOConfig.learning_rate)
    parser.add_argument("--initial-log-std", type=float, default=PPOConfig.initial_log_std)
    parser.add_argument("--seed", type=int, default=PPOConfig.seed)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = PPOConfig(
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        initial_log_std=args.initial_log_std,
        seed=args.seed,
    )
    result = train(
        scene_path=args.scene,
        robot=args.robot,
        backend=args.backend,
        project_path=args.project,
        config=config,
        device=args.device,
        gobot_pythonpath=args.gobot_pythonpath,
    )
    print(result)


if __name__ == "__main__":
    main()
