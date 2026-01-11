"""
Random seed management for reproducibility
"""

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries
    
    Args:
        seed: Random seed value
        
    Example:
        >>> set_seed(42)
        >>> # All random operations will now be deterministic
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic (slower but reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Test seeding
    print("Testing seed functionality...")
    
    set_seed(42)
    r1 = np.random.rand(5)
    t1 = torch.rand(5)
    
    set_seed(42)
    r2 = np.random.rand(5)
    t2 = torch.rand(5)
    
    print(f"NumPy arrays equal: {np.allclose(r1, r2)}")
    print(f"PyTorch tensors equal: {torch.allclose(t1, t2)}")
    print("✓ Seeding works correctly!")