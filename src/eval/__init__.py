from __future__ import annotations

from typing import Dict

import numpy as np


def evaluate(agent, env, num_episodes: int = 5) -> Dict[str, float]:
    """
    Greedy evaluation on a provided environment instance.
    """
    returns = []
    lengths = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_return = 0.0
        ep_len = 0

        while not (done or truncated):
            action, _ = agent.select_action(obs, greedy=True)
            obs, reward, done, truncated, _ = env.step(action)
            ep_return += float(reward)
            ep_len += 1

        returns.append(ep_return)
        lengths.append(ep_len)

    if not returns:
        return {"mean_return": 0.0, "eval_return_mean": 0.0}

    mean_ret = float(np.mean(returns))
    return {
        "mean_return": mean_ret,
        "eval_return_mean": mean_ret,
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
    }


__all__ = ["evaluate"]
