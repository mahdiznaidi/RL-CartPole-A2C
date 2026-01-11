"""
Critic network (value network) for A2C
"""

import torch
import torch.nn as nn
from typing import Tuple
import numpy as np
from .init_weights import init_weights


class CriticNetwork(nn.Module):
    """
    Critic network that estimates state values V(s)
    
    Architecture:
        Input (state_dim)
        → Hidden Layer 1 (hidden_size) + Activation
        → Hidden Layer 2 (hidden_size) + Activation
        → Output Layer (1) - no activation
    
    Args:
        state_dim: Dimension of state space (4 for CartPole)
        hidden_size: Number of neurons in hidden layers (default: 64)
        activation: Activation function ('tanh' or 'relu')
        
    Example:
        >>> critic = CriticNetwork(state_dim=4, hidden_size=64)
        >>> state = torch.randn(1, 4)
        >>> value = critic(state)
        >>> print(value)  # tensor([[123.45]])
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_size: int = 64,
        activation: str = "tanh"
    ):
        super().__init__()
        
        self.state_dim = state_dim
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
        self.fc3 = nn.Linear(hidden_size, 1)  # Output single value
        
        # Initialize weights
        init_weights(self.fc1, gain=np.sqrt(2))
        init_weights(self.fc2, gain=np.sqrt(2))
        init_weights(self.fc3, gain=1.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through critic network
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            values: State values of shape (batch_size, 1)
        """
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        values = self.fc3(x)
        
        return values


if __name__ == "__main__":
    # Test critic network
    print("Testing CriticNetwork...")
    
    # Create network
    critic = CriticNetwork(state_dim=4, hidden_size=64, activation="tanh")
    print(f"✓ Created critic network")
    print(f"  Parameters: {sum(p.numel() for p in critic.parameters())}")
    
    # Test forward pass
    batch_size = 3
    state = torch.randn(batch_size, 4)
    values = critic(state)
    
    print(f"✓ Forward pass works")
    print(f"  Input shape: {state.shape}")
    print(f"  Output shape: {values.shape}")
    print(f"  Output: {values.squeeze()}")
    
    # Check output is (batch_size, 1)
    assert values.shape == (batch_size, 1), f"Wrong output shape: {values.shape}"
    print(f"✓ Output shape correct")
    
    # Test single state (no batch)
    single_state = torch.randn(4)
    single_value = critic(single_state.unsqueeze(0))
    print(f"✓ Single state works: {single_value.item():.2f}")
    
    # Test gradients flow
    loss = values.mean()
    loss.backward()
    has_gradients = any(p.grad is not None for p in critic.parameters())
    assert has_gradients, "Gradients not flowing!"
    print(f"✓ Gradients flow correctly")
    
    print("\n✓ CriticNetwork tests passed!")