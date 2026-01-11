"""
Environment wrappers for CartPole
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Any


class StochasticRewardWrapper(gym.Wrapper):
    """
    Wrapper that masks rewards with a given probability
    
    Used for Agent 1+ to add stochasticity to the environment.
    The reward given to the learner is masked (set to 0) with probability p,
    but the true reward is stored for logging purposes.
    
    Args:
        env: Base environment
        mask_prob: Probability of masking reward (0.0 = no masking, 0.9 = 90% masking)
        
    Example:
        >>> env = gym.make('CartPole-v1')
        >>> env = StochasticRewardWrapper(env, mask_prob=0.9)
        >>> state, info = env.reset()
        >>> next_state, reward, done, truncated, info = env.step(0)
        >>> print(f"Masked reward: {reward}, True reward: {info['true_reward']}")
    """
    
    def __init__(self, env: gym.Env, mask_prob: float = 0.9):
        super().__init__(env)
        assert 0.0 <= mask_prob < 1.0, "mask_prob must be in [0, 1)"
        self.mask_prob = mask_prob
        self.rng = np.random.RandomState()
    
    def reset(self, **kwargs):
        """Reset environment and wrapper state"""
        # Reset the random state for reproducibility
        if 'seed' in kwargs:
            self.rng = np.random.RandomState(kwargs['seed'])
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """
        Step environment and mask reward
        
        Returns:
            next_state: Next observation
            reward: Masked reward (given to learner)
            done: Terminal flag
            truncated: Truncation flag
            info: Dict with 'true_reward' key
        """
        next_state, true_reward, done, truncated, info = self.env.step(action)
        
        # Mask reward with probability mask_prob
        if self.rng.random() < self.mask_prob:
            masked_reward = 0.0
        else:
            masked_reward = true_reward
        
        # Store true reward in info for logging
        info['true_reward'] = true_reward
        info['masked_reward'] = masked_reward
        info['reward_was_masked'] = (masked_reward == 0.0 and true_reward != 0.0)
        
        return next_state, masked_reward, done, truncated, info


class BootstrappingInfoWrapper(gym.Wrapper):
    """
    Wrapper that adds clear bootstrapping flags to info dict
    
    Helps distinguish between:
    - Termination: Episode truly ended (pole fell) → don't bootstrap
    - Truncation: Episode cut off at max steps → bootstrap
    - Normal step: Episode continues → bootstrap
    
    This is CRITICAL for correct A2C implementation!
    """
    
    def step(self, action):
        next_state, reward, done, truncated, info = self.env.step(action)
        
        # Add explicit bootstrapping flag
        # Bootstrap if: (1) not terminated OR (2) truncated
        info['should_bootstrap'] = (not done) or truncated
        
        # Also add clear flags
        info['is_terminal'] = done and not truncated
        info['is_truncated'] = truncated
        
        return next_state, reward, done, truncated, info


if __name__ == "__main__":
    # Test StochasticRewardWrapper
    print("Testing StochasticRewardWrapper...")
    
    env = gym.make('CartPole-v1')
    env = StochasticRewardWrapper(env, mask_prob=0.9)
    
    state, info = env.reset(seed=42)
    print(f"Initial state shape: {state.shape}")
    
    # Run a few steps and check masking
    masked_count = 0
    total_steps = 100
    
    for _ in range(total_steps):
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        
        if info['reward_was_masked']:
            masked_count += 1
        
        if done or truncated:
            state, info = env.reset()
    
    mask_rate = masked_count / total_steps
    print(f"Mask rate: {mask_rate:.2%} (expected ~90%)")
    print(f"✓ StochasticRewardWrapper works!")
    
    env.close()
    
    # Test BootstrappingInfoWrapper
    print("\nTesting BootstrappingInfoWrapper...")
    
    env = gym.make('CartPole-v1')
    env = BootstrappingInfoWrapper(env)
    
    state, info = env.reset(seed=42)
    
    # Run until termination or truncation
    step_count = 0
    while True:
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        step_count += 1
        
        if done or truncated:
            print(f"Episode ended at step {step_count}")
            print(f"  is_terminal: {info['is_terminal']}")
            print(f"  is_truncated: {info['is_truncated']}")
            print(f"  should_bootstrap: {info['should_bootstrap']}")
            break
    
    print("✓ BootstrappingInfoWrapper works!")
    
    env.close()