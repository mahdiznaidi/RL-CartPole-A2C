"""
File I/O utilities for saving/loading data
"""

import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Any, Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Create directory if it doesn't exist
    
    Args:
        path: Directory path
        
    Returns:
        Path: Path object
        
    Example:
        >>> ensure_dir("outputs/runs/agent0")
        PosixPath('outputs/runs/agent0')
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Save dictionary to JSON file
    
    Args:
        data: Dictionary to save
        filepath: Output file path
        
    Example:
        >>> config = {'lr': 0.001, 'gamma': 0.99}
        >>> save_json(config, 'outputs/config.json')
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved JSON to {filepath}")


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load dictionary from JSON file
    
    Args:
        filepath: Input file path
        
    Returns:
        Dict: Loaded dictionary
        
    Example:
        >>> config = load_json('outputs/config.json')
        >>> print(config['lr'])
        0.001
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data


def save_csv(
    data: List[Dict[str, Any]],
    filepath: Union[str, Path],
    fieldnames: List[str] = None
) -> None:
    """
    Save list of dictionaries to CSV file
    
    Args:
        data: List of dictionaries (each dict is a row)
        filepath: Output file path
        fieldnames: Column names (if None, uses keys from first dict)
        
    Example:
        >>> data = [
        ...     {'step': 0, 'reward': 10},
        ...     {'step': 1, 'reward': 20}
        ... ]
        >>> save_csv(data, 'outputs/log.csv')
    """
    if not data:
        print("Warning: No data to save")
        return
    
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    if fieldnames is None:
        fieldnames = list(data[0].keys())
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"✓ Saved CSV to {filepath} ({len(data)} rows)")


def append_csv(
    row: Dict[str, Any],
    filepath: Union[str, Path],
    fieldnames: List[str] = None
) -> None:
    """
    Append a single row to CSV file (creates file if doesn't exist)
    
    Args:
        row: Dictionary representing one row
        filepath: Output file path
        fieldnames: Column names
        
    Example:
        >>> row = {'step': 1000, 'loss': 0.5}
        >>> append_csv(row, 'outputs/train_log.csv')
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    file_exists = filepath.exists()
    
    if fieldnames is None:
        fieldnames = list(row.keys())
    
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row)


def load_csv(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load CSV file as list of dictionaries
    
    Args:
        filepath: Input file path
        
    Returns:
        List[Dict]: List of dictionaries (one per row)
        
    Example:
        >>> data = load_csv('outputs/log.csv')
        >>> print(data[0])
        {'step': '0', 'reward': '10'}
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    return data


if __name__ == "__main__":
    # Test I/O utilities
    print("Testing I/O utilities...")
    
    # Test directory creation
    test_dir = ensure_dir("test_outputs/test_dir")
    print(f"✓ Created directory: {test_dir}")
    
    # Test JSON
    test_data = {
        'name': 'agent0',
        'lr': 0.001,
        'gamma': 0.99
    }
    save_json(test_data, "test_outputs/test.json")
    loaded_data = load_json("test_outputs/test.json")
    assert loaded_data == test_data
    print("✓ JSON save/load works")
    
    # Test CSV
    test_rows = [
        {'step': 0, 'reward': 10.5, 'loss': 0.1},
        {'step': 1, 'reward': 20.3, 'loss': 0.05},
        {'step': 2, 'reward': 15.7, 'loss': 0.08}
    ]
    save_csv(test_rows, "test_outputs/test.csv")
    loaded_rows = load_csv("test_outputs/test.csv")
    assert len(loaded_rows) == 3
    print("✓ CSV save/load works")
    
    # Test CSV append
    append_csv({'step': 3, 'reward': 25.0, 'loss': 0.03}, "test_outputs/test.csv")
    loaded_rows = load_csv("test_outputs/test.csv")
    assert len(loaded_rows) == 4
    print("✓ CSV append works")
    
    # Cleanup
    import shutil
    shutil.rmtree("test_outputs")
    print("✓ Cleanup complete")
    
    print("✓ All I/O utilities work!")