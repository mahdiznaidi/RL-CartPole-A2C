# Create test file: tests/test_networks.py
"""
Test both networks together
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.models import ActorNetwork, CriticNetwork

def test_networks():
    print("=" * 60)
    print("TESTING NETWORKS")
    print("=" * 60)
    
    # Create networks
    actor = ActorNetwork(state_dim=4, action_dim=2, hidden_size=64)
    critic = CriticNetwork(state_dim=4, hidden_size=64)
    
    print(f"✓ Networks created")
    print(f"  Actor params: {sum(p.numel() for p in actor.parameters())}")
    print(f"  Critic params: {sum(p.numel() for p in critic.parameters())}")
    
    # Test with batch
    batch_size = 10
    states = torch.randn(batch_size, 4)
    
    # Actor
    action_probs = actor(states)
    assert action_probs.shape == (batch_size, 2)
    assert torch.allclose(action_probs.sum(dim=1), torch.ones(batch_size))
    print(f"✓ Actor output correct: {action_probs.shape}")
    
    # Critic
    values = critic(states)
    assert values.shape == (batch_size, 1)
    print(f"✓ Critic output correct: {values.shape}")
    
    # Test action sampling
    dist = actor.get_action_distribution(states)
    actions = dist.sample()
    log_probs = dist.log_prob(actions)
    
    assert actions.shape == (batch_size,)
    assert log_probs.shape == (batch_size,)
    assert torch.all((actions == 0) | (actions == 1))  # Valid actions
    print(f"✓ Action sampling works")
    print(f"  Actions: {actions}")
    print(f"  Log probs range: [{log_probs.min():.3f}, {log_probs.max():.3f}]")
    
    # Test gradients
    loss = (action_probs.mean() + values.mean())
    loss.backward()
    
    actor_has_grads = all(p.grad is not None for p in actor.parameters())
    critic_has_grads = all(p.grad is not None for p in critic.parameters())
    
    assert actor_has_grads and critic_has_grads
    print(f"✓ Gradients flow to both networks")
    
    print("\n" + "=" * 60)
    print("✓ ALL NETWORK TESTS PASSED")
    print("=" * 60)

if __name__ == "__main__":
    test_networks()