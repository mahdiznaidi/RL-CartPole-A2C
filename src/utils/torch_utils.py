"""
PyTorch utility functions
"""

import torch
import numpy as np
from typing import Union, Optional


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device (CUDA or CPU)
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
        
    Returns:
        torch.device: The device to use
        
    Example:
        >>> device = get_device()
        >>> print(device)
        device(type='cuda', index=0)  # if CUDA available
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def to_tensor(
    x: Union[np.ndarray, list, float],
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Convert numpy array, list, or scalar to PyTorch tensor
    
    Args:
        x: Input data
        device: Target device
        dtype: Target dtype
        
    Returns:
        torch.Tensor: Converted tensor
        
    Example:
        >>> device = torch.device('cpu')
        >>> x = np.array([1, 2, 3])
        >>> t = to_tensor(x, device)
        >>> print(t)
        tensor([1., 2., 3.])
    """
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    
    if isinstance(x, (int, float)):
        x = [x]
    
    return torch.tensor(x, device=device, dtype=dtype)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array
    
    Args:
        x: Input tensor
        
    Returns:
        np.ndarray: Converted array
        
    Example:
        >>> t = torch.tensor([1., 2., 3.])
        >>> arr = to_numpy(t)
        >>> print(arr)
        [1. 2. 3.]
    """
    return x.detach().cpu().numpy()


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
        
    Example:
        >>> model = torch.nn.Linear(10, 5)
        >>> print(count_parameters(model))
        55  # 10*5 weights + 5 biases
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test utilities
    print("Testing PyTorch utilities...")
    
    device = get_device()
    
    # Test to_tensor
    x_np = np.array([1, 2, 3])
    x_tensor = to_tensor(x_np, device)
    print(f"✓ to_tensor: {x_tensor}")
    
    # Test to_numpy
    y_tensor = torch.tensor([4., 5., 6.])
    y_np = to_numpy(y_tensor)
    print(f"✓ to_numpy: {y_np}")
    
    # Test count_parameters
    model = torch.nn.Linear(10, 5)
    n_params = count_parameters(model)
    print(f"✓ count_parameters: {n_params}")
    
    print("✓ All PyTorch utilities work!")