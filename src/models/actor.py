"""
Actor network (policy network) for A2C
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np
from .init_weights import init_weights


class ActorNetwork(nn.Module):
    """
    Actor network that outputs action probabilities
    
    Architecture:
        Input (state_dim) 
        → Hidden Layer 1 (hidden_size) + Activation
        → Hidden Layer 2 (hidden_size) + Activation
        → Output Layer (action_dim) + Softmax
    
    Args:
        state_dim: Dimension of state space (4 for CartPole)
        action_dim: Dimension of action space (2 for CartPole)
        hidden_size: Number of neurons in hidden layers (default: 64)
        activation: Activation function ('tanh' or 'relu')
        
    Example:
        >>> actor = ActorNetwork(state_dim=4, action_dim=2, hidden_size=64)
        >>> state = torch.randn(1, 4)
        >>> action_probs = actor(state)
        >>> print(action_probs)  # tensor([[0.52, 0.48]])
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 64,
        activation: str = "tanh"
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        
        # Choose activation function
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
        
        # Initialize weights
        init_weights(self.fc1, gain=np.sqrt(2))
        init_weights(self.fc2, gain=np.sqrt(2))
        init_weights(self.fc3, gain=0.01)  # Small gain for output layer
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through actor network
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            action_probs: Action probabilities of shape (batch_size, action_dim)
                         Probabilities sum to 1 across action dimension
        """
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        
        # Softmax to get probabilities
        action_probs = F.softmax(x, dim=-1)
        
        return action_probs
    
    def get_action_distribution(self, state: torch.Tensor) -> torch.distributions.Categorical:
        """
        Get categorical distribution over actions
        
        Useful for sampling actions and computing log probabilities
        
        Args:
            state: State tensor
            
        Returns:
            Categorical distribution over actions
            
        Example:
            >>> dist = actor.get_action_distribution(state)
            >>> action = dist.sample()
            >>> log_prob = dist.log_prob(action)
        """
        action_probs = self.forward(state)
        return torch.distributions.Categorical(action_probs)


if __name__ == "__main__":
    # Test actor network
    print("Testing ActorNetwork...")
    
    # Create network
    actor = ActorNetwork(state_dim=4, action_dim=2, hidden_size=64, activation="tanh")
    print(f"✓ Created actor network")
    print(f"  Parameters: {sum(p.numel() for p in actor.parameters())}")
    
    # Test forward pass
    batch_size = 3
    state = torch.randn(batch_size, 4)
    action_probs = actor(state)
    
    print(f"✓ Forward pass works")
    print(f"  Input shape: {state.shape}")
    print(f"  Output shape: {action_probs.shape}")
    print(f"  Output: {action_probs}")
    
    # Check probabilities sum to 1
    prob_sums = action_probs.sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones(batch_size)), "Probs don't sum to 1!"
    print(f"✓ Probabilities sum to 1: {prob_sums}")
    
    # Test action distribution
    dist = actor.get_action_distribution(state)
    actions = dist.sample()
    log_probs = dist.log_prob(actions)
    
    print(f"✓ Action distribution works")
    print(f"  Sampled actions: {actions}")
    print(f"  Log probs: {log_probs}")
    
    # Test single state (no batch)
    single_state = torch.randn(4)
    single_probs = actor(single_state.unsqueeze(0))
    print(f"✓ Single state works: {single_probs.squeeze()}")
    
    print("\n✓ ActorNetwork tests passed!")