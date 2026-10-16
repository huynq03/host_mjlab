"""Resolved original configuration, without legged_gym dependencies."""

from types import SimpleNamespace

from .original_config import DATA


def namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: namespace(v) for k, v in value.items()})
    return value


CFG = namespace(DATA["env"])
GROUPS = tuple(CFG.rewards.reward_groups)
