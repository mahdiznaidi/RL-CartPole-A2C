# Create: tests/test_algo_components.py
"""
Test algorithm components work together
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from src.algo import RolloutBuffer, compute_returns_1step, compute_returns_nstep
from src.algo import compute_actor_loss, compute_critic_loss, compute_entropy


def test_full_pipeline():
    """Test complete algorithm component pipeline"""
    print("=" * 60)
    print("TESTING ALGORITHM COMPONENTS PIPELINE")
    print("=" * 60)
    
    # Simulate collecting data
    buffer = RolloutBuffer()
    
    for i in range(5):
        state = np.random.randn(4)
        action = np.random.randint(2)
        reward = 1.0
        next_state = np.random.randn(4)
        done = (i == 4)
        truncated = False
        log_prob = -0.5
        
        buffer.push(state, action, reward, next_state, done, truncated, log_prob)
    
    print(f"✓ Collected {len(buffer)} transitions")
    
    # Get data
    data = buffer.get()
    
    # Compute values (simulate critic outputs)
    values = torch.randn(5) * 10 + 100  # Around 100
    next_values = torch.randn(5) * 10 + 100
    
    # Compute returns
    targets = compute_returns_1step(
        data['rewards'],
        values,
        next_values,
        data['dones'],
        data['truncated'],
        gamma=0.99
    )
    print(f"✓ Computed 1-step returns: {targets}")
    
    # Compute advantages
    advantages = targets - values
    print(f"✓ Computed advantages: {advantages}")
    
    # Compute losses
    log_probs = torch.tensor(data['log_probs'])
    actor_loss = compute_actor_loss(log_probs, advantages)
    critic_loss = compute_critic_loss(values, targets)
    
    print(f"✓ Actor loss: {actor_loss.item():.4f}")
    print(f"✓ Critic loss: {critic_loss.item():.4f}")
    
    # Compute entropy (simulate action probs)
    action_probs = torch.softmax(torch.randn(5, 2), dim=-1)
    entropy = compute_entropy(action_probs)
    print(f"✓ Entropy: {entropy.item():.4f}")
    
    print("\n" + "=" * 60)
    print("✓ ALGORITHM COMPONENTS PIPELINE WORKS")
    print("=" * 60)


if __name__ == "__main__":
    test_full_pipeline()