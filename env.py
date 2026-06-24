"""
Environment / dataset interface.

Swap `ToyPromptEnv` for your real task. The contract an env must satisfy:

  - `sample_prompts(n)` -> list[str]               pull a batch of n prompts/tasks
  - `rollout(prompts, generate_fn)` -> list[dict]    run generate_fn over prompts, return
                                                      trajectories: [{"prompt", "response", "meta"}]

Keep this interface stable so actors/rollout_actor.py doesn't need to change
when you point the template at a new task.
"""

import random
from typing import Callable, List, Dict, Any


class ToyPromptEnv:
    """
    Minimal reference task: short instruction-following prompts with no external
    environment state. Useful for smoke-testing the full mesh end-to-end before
    swapping in a real environment (tool-use, multi-turn, game state, etc).
    """

    PROMPTS = [
        "Summarize the benefits of distributed training in one sentence.",
        "Explain what a KL penalty does in PPO, in plain language.",
        "Write a one-line docstring for a function that retries a network call.",
        "Give a short analogy for how GRPO's group baseline works.",
        "What's the difference between rollout and trainer actors in this template?",
    ]

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def sample_prompts(self, n: int) -> List[str]:
        return [self._rng.choice(self.PROMPTS) for _ in range(n)]

    def rollout(self, prompts: List[str], generate_fn: Callable[[List[str]], List[str]]) -> List[Dict[str, Any]]:
        responses = generate_fn(prompts)
        return [
            {"prompt": p, "response": r, "meta": {}}
            for p, r in zip(prompts, responses)
        ]


def build_env(config: dict):
    """Factory — swap this to return your real environment based on config."""
    return ToyPromptEnv(seed=config.get("seed", 0))
