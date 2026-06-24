"""
Launches the training run as a Lightning job on the GPU mesh sized in config.yaml.

Usage:
    python submit_job.py --config config.yaml
"""

import argparse
import yaml
import lightning as L

from train import run_training


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    mesh_cfg = config["mesh"]
    total_gpus = mesh_cfg["num_rollout_actors"] + mesh_cfg["num_trainer_gpus"]

    job = L.Job(
        name=config["logging"].get("run_name") or "torchforge-rl-run",
        entrypoint=lambda: run_training(config, dry_run=False),
        machine=L.Multi(
            [
                L.Machine(gpu=mesh_cfg["rollout_gpu_type"], count=mesh_cfg["num_rollout_actors"]),
                L.Machine(gpu=mesh_cfg["trainer_gpu_type"], count=mesh_cfg["num_trainer_gpus"]),
            ]
        ),
    )
    job.run()
    print(f"Submitted job '{job.name}' across {total_gpus} GPUs "
          f"({mesh_cfg['num_rollout_actors']}x{mesh_cfg['rollout_gpu_type']} rollout + "
          f"{mesh_cfg['num_trainer_gpus']}x{mesh_cfg['trainer_gpu_type']} trainer).")


if __name__ == "__main__":
    main()
