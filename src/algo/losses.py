"""
Loss functions for A2C
"""

import torch
import torch.nn.functional as F


def compute_actor_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor
) -> torch.Tensor:
    """
    Compute policy gradient loss for actor
    
    Formula: L = -mean(log_prob * advantage)
    
    We want to maximize: E[log π(a|s) * A(s,a)]
    PyTorch minimizes, so we negate to get loss
    
    Args:
        log_probs: Log probabilities of actions taken, shape (batch_size,)
        advantages: Advantage values, shape (batch_size,)
        
    Returns:
        actor_loss: Scalar loss value
        
    Example:
        >>> log_probs = torch.tensor([-0.5, -0.7, -0.3])
        >>> advantages = torch.tensor([2.0, -1.0, 3.0])
        >>> loss = compute_actor_loss(log_probs, advantages)
    """
    # Policy gradient: maximize E[log π * A]
    # Equivalent to minimizing: -E[log π * A]
    actor_loss = -(log_probs * advantages).mean()
    
    return actor_loss


def compute_critic_loss(
    values: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    """
    Compute mean squared error loss for critic
    
    Formula: L = mean((V(s) - target)²)
    
    Args:
        values: Predicted state values V(s), shape (batch_size,) or (batch_size, 1)
        targets: Target values (returns), shape (batch_size,)
        
    Returns:
        critic_loss: Scalar loss value
        
    Example:
        >>> values = torch.tensor([[100.0], [99.0], [98.0]])
        >>> targets = torch.tensor([102.0, 97.0, 99.0])
        >>> loss = compute_critic_loss(values, targets)
    """
    # Reshape if needed
    if values.dim() == 2:
        values = values.squeeze(-1)
    
    # MSE loss
    critic_loss = F.mse_loss(values, targets)
    
    return critic_loss


def compute_entropy(action_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of action distribution
    
    Entropy encourages exploration. Higher entropy = more random policy
    
    Formula: H = -sum(p * log(p))
    
    Args:
        action_probs: Action probabilities, shape (batch_size, action_dim)
        
    Returns:
        entropy: Mean entropy across batch
        
    Example:
        >>> probs = torch.tensor([[0.5, 0.5], [0.9, 0.1]])
        >>> entropy = compute_entropy(probs)
        >>> print(entropy)  # First has high entropy, second low
    """
    # Avoid log(0) by adding small epsilon
    log_probs = torch.log(action_probs + 1e-8)
    
    # H = -sum(p * log(p))
    entropy = -(action_probs * log_probs).sum(dim=-1).mean()
    
    return entropy


if __name__ == "__main__":
    # Test actor loss
    print("Testing actor loss...")
    
    log_probs = torch.tensor([-0.5, -0.7, -0.3])
    advantages = torch.tensor([2.0, -1.0, 3.0])
    
    actor_loss = compute_actor_loss(log_probs, advantages)
    print(f"✓ Actor loss: {actor_loss.item():.4f}")
    
    # When advantage is positive, we want to increase log_prob (decrease negative loss)
    # When advantage is negative, we want to decrease log_prob (increase negative loss)
    expected = -((-0.5*2.0) + (-0.7*-1.0) + (-0.3*3.0)) / 3
    assert torch.allclose(actor_loss, torch.tensor(expected)), \
        f"Wrong actor loss: {actor_loss} vs {expected}"
    print(f"  Expected: {expected:.4f}")
    
    # Test critic loss
    print("\nTesting critic loss...")
    
    values = torch.tensor([[100.0], [99.0], [98.0]])
    targets = torch.tensor([102.0, 97.0, 99.0])
    
    critic_loss = compute_critic_loss(values, targets)
    print(f"✓ Critic loss: {critic_loss.item():.4f}")
    
    # MSE = mean((v - t)²)
    expected = ((100-102)**2 + (99-97)**2 + (98-99)**2) / 3
    assert torch.allclose(critic_loss, torch.tensor(expected)), \
        f"Wrong critic loss: {critic_loss} vs {expected}"
    print(f"  Expected: {expected:.4f}")
    
    # Test entropy
    print("\nTesting entropy...")
    
    # Uniform distribution (high entropy)
    uniform_probs = torch.tensor([[0.5, 0.5]])
    uniform_entropy = compute_entropy(uniform_probs)
    print(f"✓ Uniform distribution entropy: {uniform_entropy.item():.4f}")
    
    # Deterministic distribution (low entropy)
    deterministic_probs = torch.tensor([[0.99, 0.01]])
    deterministic_entropy = compute_entropy(deterministic_probs)
    print(f"✓ Deterministic distribution entropy: {deterministic_entropy.item():.4f}")
    
    assert uniform_entropy > deterministic_entropy, \
        "Uniform should have higher entropy than deterministic!"
    print(f"  Uniform > Deterministic: {uniform_entropy.item():.4f} > {deterministic_entropy.item():.4f}")
    
    print("\n✓ All loss computation tests passed!")