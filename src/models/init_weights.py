"""
Weight initialization utilities
"""

import torch
import torch.nn as nn
import numpy as np


def init_weights(module: nn.Module, gain: float = 1.0) -> None:
    """
    Initialize module weights using orthogonal initialization
    
    This initialization helps with training stability in RL.
    
    Args:
        module: PyTorch module to initialize
        gain: Scaling factor for orthogonal initialization
        
    Example:
        >>> layer = nn.Linear(64, 64)
        >>> init_weights(layer, gain=np.sqrt(2))
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


if __name__ == "__main__":
    # Test weight initialization
    print("Testing weight initialization...")
    
    layer = nn.Linear(10, 5)
    print(f"Before init: {layer.weight[0, :3]}")
    
    init_weights(layer, gain=1.0)
    print(f"After init: {layer.weight[0, :3]}")
    
    print("✓ Weight initialization works")