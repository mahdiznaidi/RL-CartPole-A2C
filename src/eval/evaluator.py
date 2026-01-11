"""
Agent evaluation utilities
"""

import numpy as np
import gymnasium as gym
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class Evaluator:
    """
    Evaluate agent performance
    
    Runs agent greedily for multiple episodes and collects statistics.
    
    Args:
        env_id: Gymnasium environment ID
        n_episodes: Number of episodes to evaluate
        seed: Random seed for evaluation environment
        
    Example:
        >>> evaluator = Evaluator('CartPole-v1', n_episodes=10, seed=42)
        >>> stats = evaluator.evaluate(agent)
        >>> print(stats['mean_return'])
    """
    
    def __init__(
        self,
        env_id: str = 'CartPole-v1',
        n_episodes: int = 10,
        seed: int = 42
    ):
        self.env_id = env_id
        self.n_episodes = n_episodes
        self.seed = seed
        
        # Create evaluation environment
        self.env = gym.make(env_id)
        self.env.reset(seed=seed)
    
    def evaluate(
        self,
        agent,
        record_trajectory: bool = False
    ) -> Dict[str, any]:
        """
        Evaluate agent for n_episodes
        
        Args:
            agent: A2C agent to evaluate
            record_trajectory: If True, record one full trajectory
            
        Returns:
            stats: Dictionary with evaluation statistics
                - mean_return: Mean episode return
                - std_return: Std of episode returns
                - min_return: Min episode return
                - max_return: Max episode return
                - returns: List of all episode returns
                - trajectory: (Optional) Full trajectory data
        """
        episode_returns = []
        episode_lengths = []
        trajectory = None
        
        for ep in range(self.n_episodes):
            obs, _ = self.env.reset()
            episode_return = 0
            episode_length = 0
            done = False
            
            # Record trajectory for first episode if requested
            if record_trajectory and ep == 0:
                traj_states = []
                traj_actions = []
                traj_rewards = []
                traj_values = []
            
            while not done:
                # Select action greedily
                action, _ = agent.select_action(obs, greedy=True)
                
                # Record trajectory data
                if record_trajectory and ep == 0:
                    traj_states.append(obs.copy())
                    traj_actions.append(action)
                    value = agent.get_value(obs)
                    traj_values.append(value)
                
                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                episode_return += reward
                episode_length += 1
                
                if record_trajectory and ep == 0:
                    traj_rewards.append(reward)
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            
            # Save trajectory from first episode
            if record_trajectory and ep == 0:
                trajectory = {
                    'states': np.array(traj_states),
                    'actions': np.array(traj_actions),
                    'rewards': np.array(traj_rewards),
                    'values': np.array(traj_values),
                    'return': episode_return,
                    'length': episode_length
                }
        
        # Compute statistics
        stats = {
            'mean_return': float(np.mean(episode_returns)),
            'std_return': float(np.std(episode_returns)),
            'min_return': float(np.min(episode_returns)),
            'max_return': float(np.max(episode_returns)),
            'mean_length': float(np.mean(episode_lengths)),
            'returns': episode_returns,
            'lengths': episode_lengths
        }
        
        if trajectory is not None:
            stats['trajectory'] = trajectory
        
        return stats
    
    def close(self):
        """Close evaluation environment"""
        self.env.close()


def evaluate_agent(
    agent,
    env_id: str = 'CartPole-v1',
    n_episodes: int = 10,
    seed: int = 42,
    record_trajectory: bool = False
) -> Dict[str, any]:
    """
    Convenience function to evaluate an agent
    
    Args:
        agent: A2C agent
        env_id: Environment ID
        n_episodes: Number of evaluation episodes
        seed: Random seed
        record_trajectory: Whether to record trajectory
        
    Returns:
        Evaluation statistics
        
    Example:
        >>> stats = evaluate_agent(agent, n_episodes=10)
        >>> print(f"Mean return: {stats['mean_return']:.2f}")
    """
    evaluator = Evaluator(env_id, n_episodes, seed)
    stats = evaluator.evaluate(agent, record_trajectory)
    evaluator.close()
    return stats


if __name__ == "__main__":
    # Test evaluator
    print("Testing evaluator...")
    
    import sys
    sys.path.insert(0, '..')
    
    from src.config import get_agent_config
    from src.algo.a2c_agent import A2CAgent
    
    # Create agent
    config = get_agent_config(0, seed=42)
    agent = A2CAgent(state_dim=4, action_dim=2, config=config)
    
    # Evaluate
    print("\n=== Evaluating untrained agent ===")
    stats = evaluate_agent(agent, n_episodes=5, record_trajectory=True)
    
    print(f"✓ Mean return: {stats['mean_return']:.2f}")
    print(f"✓ Std return: {stats['std_return']:.2f}")
    print(f"✓ Min/Max: {stats['min_return']:.0f} / {stats['max_return']:.0f}")
    print(f"✓ Returns: {stats['returns']}")
    
    if 'trajectory' in stats:
        traj = stats['trajectory']
        print(f"\n✓ Trajectory recorded:")
        print(f"  Length: {traj['length']}")
        print(f"  States shape: {traj['states'].shape}")
        print(f"  Values shape: {traj['values'].shape}")
        print(f"  First 5 values: {traj['values'][:5]}")
    
    print("\n✓ Evaluator tests passed!")