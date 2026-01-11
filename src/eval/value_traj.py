"""
Value trajectory extraction and saving
"""

import numpy as np
from pathlib import Path
from typing import Dict, List
import csv


def extract_value_trajectory(
    agent,
    env,
    max_steps: int = 500
) -> Dict[str, np.ndarray]:
    """
    Extract value function along a trajectory
    
    Args:
        agent: A2C agent
        env: Gymnasium environment
        max_steps: Maximum steps to run
        
    Returns:
        trajectory: Dictionary with states, values, rewards, etc.
        
    Example:
        >>> import gymnasium as gym
        >>> env = gym.make('CartPole-v1')
        >>> traj = extract_value_trajectory(agent, env)
        >>> print(traj['values'])
    """
    obs, _ = env.reset()
    
    states = []
    actions = []
    rewards = []
    values = []
    
    done = False
    step = 0
    
    while not done and step < max_steps:
        # Get value for current state
        value = agent.get_value(obs)
        values.append(value)
        states.append(obs.copy())
        
        # Select action greedily
        action, _ = agent.select_action(obs, greedy=True)
        actions.append(action)
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
        
        step += 1
    
    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'values': np.array(values),
        'length': len(states),
        'total_return': sum(rewards)
    }


def save_value_trajectory(
    trajectory: Dict[str, np.ndarray],
    filepath: Path,
    step: int
):
    """
    Save value trajectory to CSV
    
    Args:
        trajectory: Trajectory dictionary
        filepath: Output file path
        step: Training step number (for filename)
        
    Example:
        >>> traj = extract_value_trajectory(agent, env)
        >>> save_value_trajectory(traj, Path('outputs/runs/agent0/seed0'), step=20000)
    """
    # Create filename with step number
    filepath = Path(filepath)
    filename = filepath / f"value_traj_step_{step}.csv"
    
    # Ensure directory exists
    filepath.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    data = []
    for i in range(trajectory['length']):
        row = {
            'step_in_episode': i,
            'value': trajectory['values'][i],
            'reward': trajectory['rewards'][i] if i < len(trajectory['rewards']) else 0,
            'action': trajectory['actions'][i]
        }
        # Add state components
        for j, s in enumerate(trajectory['states'][i]):
            row[f'state_{j}'] = s
        
        data.append(row)
    
    # Save to CSV
    if data:
        fieldnames = list(data[0].keys())
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        print(f"✓ Saved value trajectory to {filename}")


if __name__ == "__main__":
    # Test value trajectory
    print("Testing value trajectory...")
    
    import sys
    sys.path.insert(0, '..')
    
    import gymnasium as gym
    from src.config import get_agent_config
    from src.algo.a2c_agent import A2CAgent
    
    # Create agent and env
    config = get_agent_config(0, seed=42)
    agent = A2CAgent(state_dim=4, action_dim=2, config=config)
    env = gym.make('CartPole-v1')
    
    # Extract trajectory
    traj = extract_value_trajectory(agent, env)
    
    print(f"✓ Trajectory length: {traj['length']}")
    print(f"✓ Total return: {traj['total_return']}")
    print(f"✓ Values shape: {traj['values'].shape}")
    print(f"✓ First 5 values: {traj['values'][:5]}")
    
    # Save trajectory
    from pathlib import Path
    save_value_trajectory(traj, Path('test_outputs'), step=0)
    
    # Cleanup
    import shutil
    shutil.rmtree('test_outputs')
    env.close()
    
    print("\n✓ Value trajectory tests passed!")