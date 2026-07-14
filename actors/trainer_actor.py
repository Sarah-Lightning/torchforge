"""
TrainerActor: computes the PPO/GRPO loss, takes the optimizer step, and
broadcasts updated policy weights back out to rollout actors.

This is the only actor that holds gradients / optimizer state. Rollout and
reward actors stay stateless with respect to training -- they only ever read
the latest broadcast weights.
"""

from typing import List, Dict, Any

from torchforge import Actor

from utils.advantages import compute_gae, compute_group_relative_advantages


class TrainerActor(Actor):
    def setup(self, config: dict):
        self.config = config
        self.algo = config["algo"]
        self.train_cfg = config["training"]
        self._version = 0
        self._model = self._load_initial_policy()
        self._ref_model = self._load_reference_policy() if self.algo == "ppo" else None
        self._optimizer = self._build_optimizer()

    # -- weight handle API consumed by RolloutActor --
    def current_version(self) -> int:
        return self._version

    def get_latest_weights(self):
        return self._model  # replace with model.state_dict() in a real implementation

    # -- core training step --
    def step(self, trajectories: List[Dict[str, Any]], rewards: List[float]) -> Dict[str, float]:
        if self.algo == "ppo":
            advantages = compute_gae(
                rewards,
                gamma=self.train_cfg["gamma"],
                lam=self.train_cfg["gae_lambda"],
            )
            loss, metrics = self._ppo_loss(trajectories, advantages)
        elif self.algo == "grpo":
            advantages = compute_group_relative_advantages(
                rewards,
                group_size=self.train_cfg["group_size"],
            )
            loss, metrics = self._grpo_loss(trajectories, advantages)
        else:
            raise ValueError(f"Unknown algo: {self.algo}")

        self._optimizer_step(loss)
        self._version += 1
        metrics["weights_version"] = self._version
        return metrics

    # -- internals (stubbed; wire up to your actual model/optimizer) --
    def _load_initial_policy(self):
        return self.config["policy"]["checkpoint"]

    def _load_reference_policy(self):
        return self.config["policy"].get("reference_checkpoint")

    def _build_optimizer(self):
        return {"lr": self.train_cfg["lr"]}  # placeholder

    def _ppo_loss(self, trajectories, advantages):
        kl_coef = self.train_cfg["kl_coef"]
        clip_range = self.train_cfg["clip_range"]
        # Replace with real PPO clipped-objective computation against self._ref_model.
        mean_adv = sum(advantages) / max(len(advantages), 1)
        loss = -mean_adv + kl_coef * 0.0  # placeholder
        return loss, {"loss": loss, "mean_advantage": mean_adv, "clip_range": clip_range}

    def _grpo_loss(self, trajectories, advantages):
        # Replace with real group-relative policy gradient computation.
        mean_adv = sum(advantages) / max(len(advantages), 1)
        loss = -mean_adv
        return loss, {"loss": loss, "mean_advantage": mean_adv}

    def _optimizer_step(self, loss):
        # Replace with real backward() + optimizer.step() + zero_grad()
        pass

    def save_checkpoint(self, out_dir: str):
        # Replace with real torch.save(...) of model/optimizer state
        path = f"{out_dir}/step_{self._version}.pt"
        return path
