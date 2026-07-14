"""
Main entrypoint: builds the ActorMesh and runs the async PPO/GRPO training loop.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --steps 50 --num-rollout-actors 2 --dry-run
"""

import argparse
import yaml

from torchforge import ActorMesh  # provided by the torchforge package

from actors import RolloutActor, RewardActor, TrainerActor
from utils.logging_utils import MetricsLogger


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--steps", type=int, default=None, help="Override training.steps")
    p.add_argument("--num-rollout-actors", type=int, default=None, help="Override mesh.num_rollout_actors")
    p.add_argument("--dry-run", action="store_true", help="Run a couple of steps locally, skip checkpointing")
    return p.parse_args()


def apply_overrides(config: dict, args) -> dict:
    if args.steps is not None:
        config["training"]["steps"] = args.steps
    if args.num_rollout_actors is not None:
        config["mesh"]["num_rollout_actors"] = args.num_rollout_actors
    return config


def build_mesh(config: dict) -> ActorMesh:
    mesh = ActorMesh()

    trainer = mesh.spawn(
        TrainerActor,
        gpus=config["mesh"]["num_trainer_gpus"],
        gpu_type=config["mesh"]["trainer_gpu_type"],
        config=config,
    )

    reward = mesh.spawn(RewardActor, config=config)

    rollout_actors = [
        mesh.spawn(
            RolloutActor,
            gpus=1,
            gpu_type=config["mesh"]["rollout_gpu_type"],
            config=config,
            policy_handle=trainer,
            env_config=config.get("training", {}),
        )
        for _ in range(config["mesh"]["num_rollout_actors"])
    ]

    return mesh, trainer, reward, rollout_actors


def run_training(config: dict, dry_run: bool = False):
    mesh, trainer, reward, rollout_actors = build_mesh(config)
    logger = MetricsLogger(config["logging"]["log_dir"], config["logging"].get("run_name"))

    steps = config["training"]["steps"]
    batch_size = config["training"]["batch_size"]
    per_actor_batch = max(batch_size // len(rollout_actors), 1)

    for step in range(steps):
        # 1. Pull trajectories from each rollout actor (parallel, async under the hood)
        trajectories = []
        for actor in rollout_actors:
            trajectories.extend(actor.collect_batch(per_actor_batch))

        # 2. Score trajectories
        rewards = reward.score(trajectories)

        # 3. Trainer consumes trajectories + rewards, updates policy, bumps weights_version
        metrics = trainer.step(trajectories, rewards)
        logger.log(step, metrics)

        # 4. Checkpoint on cadence (skipped in dry-run smoke tests)
        if not dry_run and (step + 1) % config["checkpointing"]["save_every"] == 0:
            trainer.save_checkpoint(config["checkpointing"]["out_dir"])

        if dry_run and step >= 4:
            print("Dry run complete (5 steps) -- mesh and loop wiring look healthy.")
            break

    mesh.teardown()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args)
    run_training(config, dry_run=args.dry_run)
