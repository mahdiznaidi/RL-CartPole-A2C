"""
Return computation for A2C

This is CRITICAL - correct bootstrapping is essential!
"""

import torch
from typing import List, Dict


def compute_returns_1step(
    rewards: List[float],
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: List[bool],
    truncated: List[bool],
    gamma: float
) -> torch.Tensor:
    """
    Compute 1-step returns (TD targets) for A2C
    
    CRITICAL: Must handle bootstrapping correctly!
    - If terminal (done=True, truncated=False): target = reward (no bootstrap)
    - If truncated (truncated=True): target = reward + gamma * V(s') (bootstrap)
    - Normal step: target = reward + gamma * V(s') (bootstrap)
    
    Formula:
        target = r + gamma * V(s') * (1 - terminal)
        
    Where terminal = done AND (NOT truncated)
    
    Args:
        rewards: List of rewards [r1, r2, ..., rn]
        values: Current state values V(s), shape (n,)
        next_values: Next state values V(s'), shape (n,)
        dones: List of done flags
        truncated: List of truncation flags
        gamma: Discount factor
        
    Returns:
        targets: TD targets, shape (n,)
        
    Example:
        >>> rewards = [1.0, 1.0, 1.0]
        >>> values = torch.tensor([100.0, 99.0, 98.0])
        >>> next_values = torch.tensor([99.0, 98.0, 0.0])  # Last is terminal
        >>> dones = [False, False, True]
        >>> truncated = [False, False, False]
        >>> targets = compute_returns_1step(rewards, values, next_values, 
        ...                                  dones, truncated, gamma=0.99)
    """
    n = len(rewards)
    targets = torch.zeros(n)
    
    for i in range(n):
        reward = rewards[i]
        next_value = next_values[i].item()
        
        # CRITICAL: Determine if this is a terminal state
        is_terminal = dones[i] and not truncated[i]
        
        if is_terminal:
            # Terminal state: no future value
            target = reward
        else:
            # Normal or truncated: bootstrap from next value
            target = reward + gamma * next_value
        
        targets[i] = target
    
    return targets


def compute_returns_nstep(
    rewards: List[float],
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: List[bool],
    truncated: List[bool],
    gamma: float,
    n_steps: int
) -> torch.Tensor:
    """
    Compute n-step returns for A2C
    
    For each position i, compute return using up to n future rewards:
    - Position 0: uses rewards[0:n] + bootstrap from values[n]
    - Position 1: uses rewards[1:n] + bootstrap from values[n]
    - ...
    - Position n-1: uses rewards[n-1] + bootstrap from next_values[n-1]
    
    CRITICAL: Handle episode boundaries correctly!
    - Termination: stop accumulating rewards, no bootstrap
    - Truncation: stop accumulating rewards, bootstrap from next value
    
    Args:
        rewards: List of rewards collected in this rollout
        values: Current state values V(s)
        next_values: Next state values V(s')
        dones: Done flags
        truncated: Truncation flags
        gamma: Discount factor
        n_steps: Number of steps to look ahead
        
    Returns:
        targets: N-step targets for each position
        
    Example with n=3:
        rewards = [1, 1, 1, 1, 1]
        Position 0: target = r[0] + γ*r[1] + γ²*r[2] + γ³*V(s[3])
        Position 1: target = r[1] + γ*r[2] + γ²*V(s[3])
        Position 2: target = r[2] + γ*V(s[3])
    """
    num_steps = len(rewards)
    targets = torch.zeros(num_steps)
    
    for i in range(num_steps):
        # Compute n-step return starting from position i
        G = 0.0
        discount = 1.0
        bootstrap_value = None

        # Look ahead up to n steps (or until end of data)
        steps_ahead = min(n_steps, num_steps - i)

        # Accumulate discounted rewards
        for j in range(steps_ahead):
            idx = i + j
            G += discount * rewards[idx]
            discount *= gamma

            # Truncation: stop accumulating but bootstrap from next value
            if truncated[idx]:
                bootstrap_value = next_values[idx].item()
                break

            # Termination: stop accumulating, no bootstrap
            if dones[idx]:
                bootstrap_value = None
                break

        # Bootstrap if we didn't hit a terminal state
        if bootstrap_value is None and (j == steps_ahead - 1) and not (dones[idx] and not truncated[idx]):
            if i + steps_ahead < num_steps:
                # Bootstrap from value at position i + steps_ahead
                bootstrap_value = values[i + steps_ahead].item()
            else:
                # At the end of rollout, use next_value of last step if not terminal
                if dones[-1] and not truncated[-1]:
                    bootstrap_value = None
                else:
                    bootstrap_value = next_values[-1].item()

        if bootstrap_value is not None:
            G += discount * bootstrap_value

        targets[i] = G
    
    return targets


def compute_returns_nstep_grouped(
    rewards: List[float],
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: List[bool],
    truncated: List[bool],
    gamma: float,
    n_steps: int,
    env_ids: List[int],
) -> torch.Tensor:
    """
    Compute n-step returns grouped per environment.

    This avoids mixing transitions from different workers when K>1.
    """
    num_steps = len(rewards)
    targets = torch.zeros(num_steps)

    env_to_indices: Dict[int, List[int]] = {}
    for idx, env_id in enumerate(env_ids):
        env_to_indices.setdefault(env_id, []).append(idx)

    for env_id, indices in env_to_indices.items():
        seq_rewards = [rewards[i] for i in indices]
        seq_values = values[indices]
        seq_next_values = next_values[indices]
        seq_dones = [dones[i] for i in indices]
        seq_truncated = [truncated[i] for i in indices]

        seq_targets = compute_returns_nstep(
            rewards=seq_rewards,
            values=seq_values,
            next_values=seq_next_values,
            dones=seq_dones,
            truncated=seq_truncated,
            gamma=gamma,
            n_steps=n_steps,
        )
        for local_idx, global_idx in enumerate(indices):
            targets[global_idx] = seq_targets[local_idx]

    return targets


if __name__ == "__main__":
    # Test 1-step returns
    print("Testing 1-step returns...")
    
    # Normal steps (no termination)
    rewards = [1.0, 1.0, 1.0]
    values = torch.tensor([100.0, 99.0, 98.0])
    next_values = torch.tensor([99.0, 98.0, 97.0])
    dones = [False, False, False]
    truncated = [False, False, False]
    
    targets = compute_returns_1step(rewards, values, next_values, dones, truncated, gamma=0.99)
    expected = torch.tensor([1 + 0.99*99, 1 + 0.99*98, 1 + 0.99*97])
    assert torch.allclose(targets, expected), f"Wrong targets: {targets} vs {expected}"
    print(f"✓ Normal steps: {targets}")
    
    # Terminal step
    rewards = [1.0, 1.0, 1.0]
    values = torch.tensor([100.0, 99.0, 98.0])
    next_values = torch.tensor([99.0, 98.0, 0.0])
    dones = [False, False, True]  # Last is terminal
    truncated = [False, False, False]
    
    targets = compute_returns_1step(rewards, values, next_values, dones, truncated, gamma=0.99)
    assert targets[2] == 1.0, f"Terminal step should have target=reward: {targets[2]}"
    print(f"✓ Terminal step: {targets}")
    
    # Truncated step (should bootstrap!)
    rewards = [1.0, 1.0, 1.0]
    values = torch.tensor([100.0, 99.0, 98.0])
    next_values = torch.tensor([99.0, 98.0, 97.0])
    dones = [False, False, True]  # Done...
    truncated = [False, False, True]  # ...but truncated!
    
    targets = compute_returns_1step(rewards, values, next_values, dones, truncated, gamma=0.99)
    expected_last = 1.0 + 0.99 * 97.0
    assert torch.allclose(targets[2], torch.tensor(expected_last)), \
        f"Truncated step should bootstrap: {targets[2]} vs {expected_last}"
    print(f"✓ Truncated step (bootstraps): {targets}")
    
    print("\nTesting n-step returns...")
    
    # Simple n-step case
    rewards = [1.0, 1.0, 1.0, 1.0, 1.0]
    values = torch.tensor([100.0, 99.0, 98.0, 97.0, 96.0])
    next_values = torch.tensor([99.0, 98.0, 97.0, 96.0, 95.0])
    dones = [False] * 5
    truncated = [False] * 5
    
    targets_3step = compute_returns_nstep(rewards, values, next_values, dones, truncated, 
                                          gamma=0.99, n_steps=3)
    print(f"✓ 3-step returns: {targets_3step}")
    
    # First position should use 3 rewards + bootstrap
    expected_first = 1.0 + 0.99*1.0 + 0.99**2*1.0 + 0.99**3*values[3].item()
    assert torch.allclose(targets_3step[0], torch.tensor(expected_first)), \
        f"Wrong 3-step return: {targets_3step[0]} vs {expected_first}"
    print(f"  First position uses 3 rewards: {targets_3step[0]:.2f}")
    
    print("\n✓ All return computation tests passed!")
