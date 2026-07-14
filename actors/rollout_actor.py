"""
RolloutActor: generates trajectories from the current policy weights.

Lives for the whole training run. Periodically pulls the latest policy
weights from the trainer actor (see `mesh.weight_sync_every` in config.yaml)
and keeps sampling against them while the trainer computes the next update.
"""

from typing import List, Dict, Any

from torchforge import Actor  # provided by the torchforge package

from env import build_env


class RolloutActor(Actor):
    def setup(self, config: dict, policy_handle, env_config: dict):
        """
        Args:
            config: full run config (dict from config.yaml)
            policy_handle: handle to the trainer actor's policy weights,
                supports `.get_latest_weights()` and `.current_version()`
            env_config: the `training`/task-specific section of config.yaml
        """
        self.config = config
        self.policy_handle = policy_handle
        self.env = build_env(env_config)
        self._weights_version = -1
        self._model = None  # lazily loaded on first generate() call

    def _maybe_refresh_weights(self):
        latest_version = self.policy_handle.current_version()
        if latest_version != self._weights_version:
            weights = self.policy_handle.get_latest_weights()
            self._load_weights(weights)
            self._weights_version = latest_version

    def _load_weights(self, weights):
        # Replace with your actual model load (e.g. model.load_state_dict(weights))
        self._model = weights

    def _generate(self, prompts: List[str]) -> List[str]:
        # Replace with real generation (e.g. HF generate / vLLM call) using self._model
        max_new_tokens = self.config["policy"]["max_new_tokens"]
        temperature = self.config["policy"]["temperature"]
        return [f"[stub response, T={temperature}, max_tok={max_new_tokens}] {p[:40]}" for p in prompts]

    def collect_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Called by the trainer actor (or a coordinator) to pull a batch of trajectories."""
        self._maybe_refresh_weights()
        prompts = self.env.sample_prompts(batch_size)
        trajectories = self.env.rollout(prompts, self._generate)
        for t in trajectories:
            t["weights_version"] = self._weights_version
        return trajectories
