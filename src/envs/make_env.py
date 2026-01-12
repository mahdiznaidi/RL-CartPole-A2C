"""
Environment factory functions
"""

import gymnasium as gym
from gymnasium.vector import AutoresetMode
from typing import Optional, Callable
from .wrappers import StochasticRewardWrapper, BootstrappingInfoWrapper


def make_env(
    seed: int = 42,
    reward_mask_prob: float = 0.0,
    render_mode: Optional[str] = None
) -> gym.Env:
    """
    Create a single CartPole environment with optional wrappers
    
    Args:
        seed: Random seed for environment
        reward_mask_prob: Probability of masking rewards (0.0 = no masking)
        render_mode: Rendering mode ('human', 'rgb_array', or None)
        
    Returns:
        gym.Env: Configured environment
        
    Example:
        >>> env = make_env(seed=42, reward_mask_prob=0.9)
        >>> state, info = env.reset()
    """
    # Create base environment
    env = gym.make('CartPole-v1', render_mode=render_mode)
    
    # Add bootstrapping info wrapper (always add this!)
    env = BootstrappingInfoWrapper(env)
    
    # Add stochastic reward wrapper if needed
    if reward_mask_prob > 0.0:
        env = StochasticRewardWrapper(env, mask_prob=reward_mask_prob)
    
    # Set seed
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    
    return env


def make_vec_env(
    num_envs: int = 1,
    seed: int = 42,
    reward_mask_prob: float = 0.0,
    async_envs: bool = False
) -> gym.vector.VectorEnv:
    """
    Create vectorized parallel environments
    
    Args:
        num_envs: Number of parallel environments (K)
        seed: Base random seed (each env gets seed + i)
        reward_mask_prob: Probability of masking rewards
        async_envs: Whether to use async or sync vectorization
        
    Returns:
        gym.vector.VectorEnv: Vectorized environment
        
    Example:
        >>> envs = make_vec_env(num_envs=6, seed=42, reward_mask_prob=0.9)
        >>> states, infos = envs.reset()
        >>> print(states.shape)  # (6, 4)
    """
    # Create a function that makes one environment
    def make_single_env(rank: int) -> Callable[[], gym.Env]:
        def _init():
            return make_env(
                seed=seed + rank,
                reward_mask_prob=reward_mask_prob,
                render_mode=None
            )
        return _init
    
    # Create list of env creation functions
    env_fns = [make_single_env(i) for i in range(num_envs)]
    
    # Create vectorized environment
    if async_envs and num_envs > 1:
        # Async: faster but more complex
        envs = gym.vector.AsyncVectorEnv(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)
    else:
        # Sync: simpler and deterministic
        envs = gym.vector.SyncVectorEnv(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)
    
    return envs


if __name__ == "__main__":
    # Test single environment
    print("Testing single environment creation...")
    
    env = make_env(seed=42, reward_mask_prob=0.0)
    print(f"✓ Created environment: {env}")
    
    state, info = env.reset()
    print(f"✓ State shape: {state.shape}")
    
    action = env.action_space.sample()
    next_state, reward, done, truncated, info = env.step(action)
    print(f"✓ Step works, reward: {reward}")
    print(f"✓ Info keys: {info.keys()}")
    
    env.close()
    
    # Test with stochastic rewards
    print("\nTesting with stochastic rewards...")
    
    env = make_env(seed=42, reward_mask_prob=0.9)
    state, info = env.reset()
    
    masked_count = 0
    for _ in range(100):
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        
        if 'reward_was_masked' in info and info['reward_was_masked']:
            masked_count += 1
        
        if done or truncated:
            state, info = env.reset()
    
    print(f"✓ Mask rate: {masked_count/100:.1%}")
    
    env.close()
    
    # Test vectorized environments
    print("\nTesting vectorized environments...")
    
    num_envs = 4
    envs = make_vec_env(num_envs=num_envs, seed=42, reward_mask_prob=0.0)
    
    states, infos = envs.reset()
    print(f"✓ States shape: {states.shape} (expected: ({num_envs}, 4))")
    
    actions = envs.action_space.sample()
    print(f"✓ Actions shape: {actions.shape} (expected: ({num_envs},))")
    
    next_states, rewards, dones, truncated, infos = envs.step(actions)
    print(f"✓ Next states shape: {next_states.shape}")
    print(f"✓ Rewards shape: {rewards.shape}")
    print(f"✓ Dones shape: {dones.shape}")
    
    envs.close()
    
    print("\n✓ All environment tests passed!")
