"""
Advantage estimators for PPO (GAE) and GRPO (group-relative baseline).
"""

from typing import List


def compute_gae(rewards: List[float], gamma: float = 1.0, lam: float = 0.95, values: List[float] = None) -> List[float]:
    """
    Generalized Advantage Estimation, used by PPO.

    If `values` (a value-function baseline per step) isn't provided, falls back
    to a zero baseline, i.e. advantages == discounted rewards. Wire up a real
    value head for production use.
    """
    if values is None:
        values = [0.0] * len(rewards)

    advantages = []
    gae = 0.0
    next_value = 0.0
    for r, v in zip(reversed(rewards), reversed(values)):
        delta = r + gamma * next_value - v
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        next_value = v
    return advantages


def compute_group_relative_advantages(rewards: List[float], group_size: int) -> List[float]:
    """
    GRPO-style advantage: subtract the mean reward within each group of
    `group_size` samples (e.g. samples drawn from the same prompt), so the
    baseline is the group's own average rather than a learned value function.
    """
    if len(rewards) % group_size != 0:
        raise ValueError(
            f"len(rewards)={len(rewards)} is not divisible by group_size={group_size}"
        )

    advantages = []
    for i in range(0, len(rewards), group_size):
        group = rewards[i : i + group_size]
        group_mean = sum(group) / len(group)
        advantages.extend([r - group_mean for r in group])
    return advantages
