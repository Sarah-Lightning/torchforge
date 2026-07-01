"""
RewardActor: scores trajectories produced by rollout actors.

Set `reward.mode: external` in config.yaml to call an external reward model
or rule-based service (e.g. a LitServe-hosted reward model) instead of the
built-in toy reward function. No changes needed in rollout/trainer actors
either way -- they only ever see a list of floats back from `score()`.
"""

from typing import List, Dict, Any
import requests

from torchforge import Actor


class RewardActor(Actor):
    def setup(self, config: dict):
        self.config = config["reward"]
        self.mode = self.config.get("mode", "builtin")
        self.endpoint = self.config.get("external_endpoint")

    def score(self, trajectories: List[Dict[str, Any]]) -> List[float]:
        if self.mode == "builtin":
            return [self._builtin_reward(t) for t in trajectories]
        elif self.mode == "external":
            return self._external_reward(trajectories)
        else:
            raise ValueError(f"Unknown reward mode: {self.mode}")

    def _builtin_reward(self, trajectory: Dict[str, Any]) -> float:
        """
        Toy reward: rewards longer, non-empty responses and lightly penalizes
        repetition. Replace with a real reward model or task-specific rule set.
        """
        response = trajectory.get("response", "")
        if not response.strip():
            return -1.0
        length_score = min(len(response.split()) / 50.0, 1.0)
        unique_ratio = len(set(response.split())) / max(len(response.split()), 1)
        return float(0.7 * length_score + 0.3 * unique_ratio)

    def _external_reward(self, trajectories: List[Dict[str, Any]]) -> List[float]:
        if not self.endpoint:
            raise ValueError("reward.mode is 'external' but reward.external_endpoint is not set")
        payload = {
            "items": [
                {"prompt": t["prompt"], "response": t["response"]} for t in trajectories
            ]
        }
        resp = requests.post(self.endpoint, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["scores"]
