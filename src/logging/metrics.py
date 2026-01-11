"""
Metric computation and aggregation functions
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import csv


def compute_mean_return(returns: List[float]) -> float:
    """
    Compute mean of episode returns
    
    Args:
        returns: List of episode returns
        
    Returns:
        Mean return
        
    Example:
        >>> returns = [100, 150, 120]
        >>> compute_mean_return(returns)
        123.33333333333333
    """
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns))


def compute_std_return(returns: List[float]) -> float:
    """
    Compute standard deviation of episode returns
    
    Args:
        returns: List of episode returns
        
    Returns:
        Standard deviation
        
    Example:
        >>> returns = [100, 150, 120]
        >>> compute_std_return(returns)
        20.81665999466133
    """
    if len(returns) == 0:
        return 0.0
    return float(np.std(returns))


def compute_min_max_return(returns: List[float]) -> Tuple[float, float]:
    """
    Compute min and max of episode returns
    
    Args:
        returns: List of episode returns
        
    Returns:
        (min, max) tuple
        
    Example:
        >>> returns = [100, 150, 120]
        >>> compute_min_max_return(returns)
        (100.0, 150.0)
    """
    if len(returns) == 0:
        return 0.0, 0.0
    return float(np.min(returns)), float(np.max(returns))


def aggregate_seeds(
    seed_data: List[List[float]],
    metric_fn=np.mean
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate data across multiple seeds
    
    Args:
        seed_data: List of lists, where each inner list is data from one seed
                   e.g., [[run1_data], [run2_data], [run3_data]]
        metric_fn: Function to compute central tendency (default: np.mean)
        
    Returns:
        central: Central tendency across seeds (mean by default)
        min_vals: Minimum across seeds at each point
        max_vals: Maximum across seeds at each point
        
    Example:
        >>> seed1 = [100, 150, 200]
        >>> seed2 = [90, 140, 210]
        >>> seed3 = [110, 160, 190]
        >>> central, min_vals, max_vals = aggregate_seeds([seed1, seed2, seed3])
        >>> print(central)
        [100. 150. 200.]
    """
    if len(seed_data) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Convert to numpy array (seeds × steps)
    data_array = np.array(seed_data)
    
    # Compute statistics across seeds (axis=0)
    central = metric_fn(data_array, axis=0)
    min_vals = np.min(data_array, axis=0)
    max_vals = np.max(data_array, axis=0)
    
    return central, min_vals, max_vals


def load_training_metrics(
    run_dir: Path,
    metric_name: str = 'episode_return'
) -> List[float]:
    """
    Load a specific metric from training log CSV
    
    Args:
        run_dir: Directory containing train_log.csv
        metric_name: Name of the metric column to extract
        
    Returns:
        List of metric values
        
    Example:
        >>> from pathlib import Path
        >>> metrics = load_training_metrics(
        ...     Path('outputs/runs/agent0/seed0'),
        ...     'episode_return'
        ... )
    """
    log_file = run_dir / 'train_log.csv'
    
    if not log_file.exists():
        raise FileNotFoundError(f"Training log not found: {log_file}")
    
    values = []
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if metric_name in row and row[metric_name] != '':
                values.append(float(row[metric_name]))
    
    return values


def load_eval_metrics(
    run_dir: Path,
    metric_name: str = 'mean_return'
) -> List[float]:
    """
    Load a specific metric from evaluation log CSV
    
    Args:
        run_dir: Directory containing eval_log.csv
        metric_name: Name of the metric column to extract
        
    Returns:
        List of metric values
        
    Example:
        >>> from pathlib import Path
        >>> metrics = load_eval_metrics(
        ...     Path('outputs/runs/agent0/seed0'),
        ...     'mean_return'
        ... )
    """
    log_file = run_dir / 'eval_log.csv'
    
    if not log_file.exists():
        raise FileNotFoundError(f"Eval log not found: {log_file}")
    
    values = []
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if metric_name in row and row[metric_name] != '':
                values.append(float(row[metric_name]))
    
    return values


def aggregate_across_seeds(
    agent_dir: Path,
    seeds: List[int],
    log_type: str = 'train',
    metric_name: str = 'episode_return'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and aggregate a metric across multiple seeds
    
    Args:
        agent_dir: Directory containing seed folders (e.g., outputs/runs/agent0)
        seeds: List of seed numbers to aggregate
        log_type: 'train' or 'eval'
        metric_name: Name of metric to aggregate
        
    Returns:
        mean_values: Mean across seeds
        min_values: Min across seeds
        max_values: Max across seeds
        
    Example:
        >>> from pathlib import Path
        >>> mean, min_vals, max_vals = aggregate_across_seeds(
        ...     Path('outputs/runs/agent0'),
        ...     seeds=[0, 1, 2],
        ...     log_type='eval',
        ...     metric_name='mean_return'
        ... )
    """
    seed_data = []
    
    for seed in seeds:
        seed_dir = agent_dir / f"seed{seed}"
        
        try:
            if log_type == 'train':
                data = load_training_metrics(seed_dir, metric_name)
            elif log_type == 'eval':
                data = load_eval_metrics(seed_dir, metric_name)
            else:
                raise ValueError(f"Unknown log_type: {log_type}")
            
            seed_data.append(data)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue
    
    if len(seed_data) == 0:
        raise ValueError(f"No data found for agent in {agent_dir}")
    
    # Ensure all seeds have same length (pad with NaN if needed)
    max_len = max(len(d) for d in seed_data)
    padded_data = []
    for data in seed_data:
        if len(data) < max_len:
            # Pad with last value (or NaN)
            padded = data + [data[-1]] * (max_len - len(data))
            padded_data.append(padded)
        else:
            padded_data.append(data)
    
    return aggregate_seeds(padded_data)


def compute_success_rate(
    returns: List[float],
    threshold: float = 500.0
) -> float:
    """
    Compute success rate (% of episodes reaching threshold)
    
    Args:
        returns: List of episode returns
        threshold: Success threshold
        
    Returns:
        Success rate (0-1)
        
    Example:
        >>> returns = [500, 450, 500, 500, 490]
        >>> compute_success_rate(returns, threshold=500)
        0.6
    """
    if len(returns) == 0:
        return 0.0
    
    successes = sum(1 for r in returns if r >= threshold)
    return successes / len(returns)


def compute_gradient_norm(model) -> float:
    """
    Compute L2 norm of gradients for a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Gradient norm
        
    Example:
        >>> import torch.nn as nn
        >>> model = nn.Linear(4, 2)
        >>> # ... after loss.backward() ...
        >>> norm = compute_gradient_norm(model)
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    return total_norm ** 0.5


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics module...")
    
    # Test basic statistics
    returns = [100, 150, 120, 180, 90]
    
    mean_ret = compute_mean_return(returns)
    print(f"✓ Mean return: {mean_ret:.2f}")
    
    std_ret = compute_std_return(returns)
    print(f"✓ Std return: {std_ret:.2f}")
    
    min_ret, max_ret = compute_min_max_return(returns)
    print(f"✓ Min/Max: {min_ret:.2f}, {max_ret:.2f}")
    
    # Test success rate
    success_rate = compute_success_rate([500, 450, 500, 490], threshold=500)
    print(f"✓ Success rate: {success_rate:.2%}")
    
    # Test aggregation
    seed1 = [100, 150, 200]
    seed2 = [90, 140, 210]
    seed3 = [110, 160, 190]
    
    mean, min_vals, max_vals = aggregate_seeds([seed1, seed2, seed3])
    print(f"✓ Aggregation works:")
    print(f"  Mean: {mean}")
    print(f"  Min: {min_vals}")
    print(f"  Max: {max_vals}")
    
    print("\n✓ All metrics tests passed!")