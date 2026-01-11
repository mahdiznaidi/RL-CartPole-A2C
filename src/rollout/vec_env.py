"""
Vectorized environment wrapper for parallel data collection
"""

import gymnasium as gym
import numpy as np
from typing import List, Tuple, Optional, Union
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv


def make_single_env(env_id: str, seed: int, reward_mask_prob: float = 0.0):
    """
    Create a single environment with optional reward masking
    
    Args:
        env_id: Gymnasium environment ID (e.g., 'CartPole-v1')
        seed: Random seed for this environment
        reward_mask_prob: Probability of masking rewards (0.0 = no masking)
        
    Returns:
        Function that creates the environment
        
    Example:
        >>> env_fn = make_single_env('CartPole-v1', seed=42, reward_mask_prob=0.9)
        >>> env = env_fn()
    """
    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed)
        
        # Apply reward masking wrapper if needed
        if reward_mask_prob > 0:
            from src.envs.wrappers import RewardMaskWrapper
            env = RewardMaskWrapper(env, mask_prob=reward_mask_prob)
        
        return env
    
    return _init


def make_vec_env(
    env_id: str,
    n_envs: int,
    base_seed: int = 0,
    reward_mask_prob: float = 0.0,
    async_envs: bool = False
) -> Union[SyncVectorEnv, AsyncVectorEnv]:
    """
    Create vectorized environment with K parallel workers
    
    Args:
        env_id: Gymnasium environment ID
        n_envs: Number of parallel environments (K)
        base_seed: Base seed (each env gets base_seed + i)
        reward_mask_prob: Probability of masking rewards
        async_envs: If True, use AsyncVectorEnv (parallel processes)
                    If False, use SyncVectorEnv (serial, easier to debug)
        
    Returns:
        Vectorized environment
        
    Example:
        >>> vec_env = make_vec_env('CartPole-v1', n_envs=6, base_seed=42)
        >>> obs = vec_env.reset()
        >>> print(obs.shape)  # (6, 4)
    """
    # Create list of environment creation functions
    env_fns = [
        make_single_env(env_id, base_seed + i, reward_mask_prob)
        for i in range(n_envs)
    ]
    
    # Create vectorized environment
    if async_envs:
        vec_env = AsyncVectorEnv(env_fns)
        print(f"Created AsyncVectorEnv with {n_envs} parallel workers")
    else:
        vec_env = SyncVectorEnv(env_fns)
        print(f"Created SyncVectorEnv with {n_envs} serial workers")
    
    return vec_env


class VecEnvWrapper:
    """
    Wrapper around Gymnasium's vectorized environments
    
    Provides additional tracking for episode statistics and
    easier interface for data collection.
    
    Args:
        vec_env: Gymnasium vectorized environment
        
    Example:
        >>> base_env = make_vec_env('CartPole-v1', n_envs=6)
        >>> env = VecEnvWrapper(base_env)
        >>> obs = env.reset()
    """
    
    def __init__(self, vec_env):
        self.vec_env = vec_env
        self.n_envs = vec_env.num_envs
        
        # Track episode returns
        self.episode_returns = np.zeros(self.n_envs)
        self.episode_lengths = np.zeros(self.n_envs, dtype=np.int32)
        
        # Storage for finished episodes
        self.finished_episodes = []
    
    def reset(self, seed: Optional[int] = None):
        """Reset all environments"""
        if seed is not None:
            obs, info = self.vec_env.reset(seed=seed)
        else:
            obs, info = self.vec_env.reset()
        
        self.episode_returns = np.zeros(self.n_envs)
        self.episode_lengths = np.zeros(self.n_envs, dtype=np.int32)
        
        return obs, info
    
    def step(self, actions: np.ndarray):
        """
        Step all environments
        
        Args:
            actions: Array of actions, shape (n_envs,)
            
        Returns:
            obs: Observations, shape (n_envs, obs_dim)
            rewards: Rewards, shape (n_envs,)
            terminated: Termination flags, shape (n_envs,)
            truncated: Truncation flags, shape (n_envs,)
            infos: List of info dicts
        """
        obs, rewards, terminated, truncated, infos = self.vec_env.step(actions)
        
        # Update episode tracking
        self.episode_returns += rewards
        self.episode_lengths += 1
        
        # Check for finished episodes
        dones = terminated | truncated
        
        for i in range(self.n_envs):
            if dones[i]:
                # Store episode info
                episode_info = {
                    'return': self.episode_returns[i],
                    'length': self.episode_lengths[i],
                    'terminated': terminated[i],
                    'truncated': truncated[i]
                }
                self.finished_episodes.append(episode_info)
                
                # Reset tracking for this environment
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
        
        return obs, rewards, terminated, truncated, infos
    
    def get_finished_episodes(self) -> List[dict]:
        """
        Get and clear list of finished episodes
        
        Returns:
            List of episode info dicts
        """
        episodes = self.finished_episodes.copy()
        self.finished_episodes = []
        return episodes
    
    def close(self):
        """Close all environments"""
        self.vec_env.close()


if __name__ == "__main__":
    # Test vectorized environments
    print("Testing vectorized environments...")
    
    # Test single env creation
    env_fn = make_single_env('CartPole-v1', seed=42)
    env = env_fn()
    obs, _ = env.reset()
    print(f"✓ Single env obs shape: {obs.shape}")
    env.close()
    
    # Test vectorized env
    vec_env = make_vec_env('CartPole-v1', n_envs=4, base_seed=42)
    obs, _ = vec_env.reset()
    print(f"✓ Vec env obs shape: {obs.shape}")  # Should be (4, 4)
    
    # Test stepping
    actions = np.array([0, 1, 0, 1])
    obs, rewards, terminated, truncated, infos = vec_env.step(actions)
    print(f"✓ Step works - obs: {obs.shape}, rewards: {rewards.shape}")
    
    vec_env.close()
    
    # Test wrapper
    vec_env = make_vec_env('CartPole-v1', n_envs=2, base_seed=42)
    wrapped_env = VecEnvWrapper(vec_env)
    obs, _ = wrapped_env.reset()
    print(f"✓ Wrapper reset works")
    
    # Run a few steps
    for _ in range(10):
        actions = np.array([0, 1])
        obs, rewards, terminated, truncated, infos = wrapped_env.step(actions)
    
    finished = wrapped_env.get_finished_episodes()
    print(f"✓ Wrapper tracking works - {len(finished)} episodes finished")
    
    wrapped_env.close()
    
    print("\n✓ All vectorized env tests passed!")