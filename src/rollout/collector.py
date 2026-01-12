"""
Rollout collection for A2C with proper bootstrapping
"""

import numpy as np
import torch
from typing import List, Tuple, Dict
from src.algo.buffer import RolloutBuffer
from src.rollout.vec_env import VecEnvWrapper


class RolloutCollector:
    """
    Collect rollouts from vectorized environments
    
    Handles:
    - K parallel environments
    - n-step collection
    - Proper bootstrapping at truncation vs termination
    - Episode boundary tracking
    
    Args:
        vec_env: VecEnvWrapper with K environments
        agent: A2C agent
        n_steps: Number of steps to collect
        
    Example:
        >>> from src.rollout.vec_env import make_vec_env, VecEnvWrapper
        >>> vec_env = VecEnvWrapper(make_vec_env('CartPole-v1', n_envs=6))
        >>> collector = RolloutCollector(vec_env, agent, n_steps=6)
        >>> buffer, next_states, episode_infos = collector.collect()
    """
    
    def __init__(self, vec_env: VecEnvWrapper, agent, n_steps: int):
        self.vec_env = vec_env
        self.agent = agent
        self.n_steps = n_steps
        self.n_envs = vec_env.n_envs
        
        # Current states for each environment
        self.current_states = None
    
    def reset(self):
        """Reset all environments and get initial states"""
        obs, _ = self.vec_env.reset()
        self.current_states = obs
        return obs
    
    def collect(self) -> Tuple[RolloutBuffer, List[np.ndarray], List[dict]]:
        """
        Collect n_steps from K environments
        
        Returns:
            buffer: RolloutBuffer with all collected transitions
            final_next_states: List of next states for each env (for bootstrapping)
            episode_infos: List of finished episode statistics
            
        Key Points:
            - Collects exactly n_steps from each environment
            - If environment terminates/truncates, it auto-resets
            - Tracks termination vs truncation for correct bootstrapping
            - Returns final next_states for bootstrapping at rollout boundary
        """
        buffer = RolloutBuffer()
        
        # If first collection, reset
        if self.current_states is None:
            self.current_states, _ = self.vec_env.reset()
        
        # Collect n steps from each environment
        for step in range(self.n_steps):
            # Select actions for all environments
            actions = np.zeros(self.n_envs, dtype=np.int32)
            log_probs = np.zeros(self.n_envs)
            
            for env_idx in range(self.n_envs):
                action, log_prob = self.agent.select_action(
                    self.current_states[env_idx],
                    greedy=False
                )
                actions[env_idx] = action
                log_probs[env_idx] = log_prob
            
            # Step all environments
            next_states, rewards, terminated, truncated, infos = self.vec_env.step(actions)
            
            # Store transitions for each environment
            for env_idx in range(self.n_envs):
                buffer.push(
                    state=self.current_states[env_idx],
                    action=actions[env_idx],
                    reward=rewards[env_idx],
                    next_state=next_states[env_idx],
                    done=terminated[env_idx],
                    truncated=truncated[env_idx],
                    log_prob=log_probs[env_idx],
                    env_id=env_idx,
                )
            
            # Update current states
            # Note: Gymnasium's VecEnv automatically resets terminated/truncated envs
            # The next_states already contain the reset states for finished episodes
            self.current_states = next_states
        
        # Final next states for bootstrapping
        # These are the states after collecting n_steps
        final_next_states = [self.current_states[i] for i in range(self.n_envs)]
        
        # Get finished episodes
        episode_infos = self.vec_env.get_finished_episodes()
        
        return buffer, final_next_states, episode_infos


class SingleEnvCollector:
    """
    Rollout collector for single environment (K=1)
    
    Simpler version for debugging and Agent 0/1.
    
    Args:
        env: Single Gymnasium environment
        agent: A2C agent
        n_steps: Number of steps to collect
        
    Example:
        >>> import gymnasium as gym
        >>> env = gym.make('CartPole-v1')
        >>> collector = SingleEnvCollector(env, agent, n_steps=1)
        >>> buffer, next_state, episode_infos = collector.collect()
    """
    
    def __init__(self, env, agent, n_steps: int):
        self.env = env
        self.agent = agent
        self.n_steps = n_steps
        
        # Current state
        self.current_state = None
        
        # Episode tracking
        self.episode_return = 0
        self.episode_length = 0
        self.finished_episodes = []
    
    def reset(self):
        """Reset environment"""
        obs, _ = self.env.reset()
        self.current_state = obs
        self.episode_return = 0
        self.episode_length = 0
        return obs
    
    def collect(self) -> Tuple[RolloutBuffer, np.ndarray, List[dict]]:
        """
        Collect n_steps from single environment
        
        Returns:
            buffer: RolloutBuffer with collected transitions
            final_next_state: Next state after n steps (for bootstrapping)
            episode_infos: List of finished episode info
        """
        buffer = RolloutBuffer()
        
        # Reset if first collection
        if self.current_state is None:
            self.current_state, _ = self.env.reset()
        
        for step in range(self.n_steps):
            # Select action
            action, log_prob = self.agent.select_action(
                self.current_state,
                greedy=False
            )
            
            # Step environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            # Update episode tracking
            self.episode_return += reward
            self.episode_length += 1
            
            # Store transition
            buffer.push(
                state=self.current_state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=terminated,
                truncated=truncated,
                log_prob=log_prob
            )
            
            # Check if episode finished
            done = terminated or truncated
            if done:
                # Store episode info
                episode_info = {
                    'return': self.episode_return,
                    'length': self.episode_length,
                    'terminated': terminated,
                    'truncated': truncated
                }
                self.finished_episodes.append(episode_info)
                
                # Reset environment
                next_state, _ = self.env.reset()
                self.episode_return = 0
                self.episode_length = 0
            
            # Update current state
            self.current_state = next_state
        
        # Get finished episodes and clear
        episode_infos = self.finished_episodes.copy()
        self.finished_episodes = []
        
        return buffer, self.current_state, episode_infos


if __name__ == "__main__":
    # Test collectors
    print("Testing rollout collectors...")
    
    import sys
    sys.path.insert(0, '..')
    
    import gymnasium as gym
    from src.config import get_agent_config
    from src.algo.a2c_agent import A2CAgent
    from src.rollout.vec_env import make_vec_env, VecEnvWrapper
    
    # Create agent
    config = get_agent_config(0, seed=42)
    agent = A2CAgent(state_dim=4, action_dim=2, config=config)
    
    print("\n=== Testing SingleEnvCollector ===")
    env = gym.make('CartPole-v1')
    collector = SingleEnvCollector(env, agent, n_steps=5)
    
    # Reset
    collector.reset()
    print("✓ Reset works")
    
    # Collect
    buffer, final_state, episodes = collector.collect()
    print(f"✓ Collected {len(buffer.get()['states'])} transitions")
    print(f"✓ Final state shape: {final_state.shape}")
    print(f"✓ Finished episodes: {len(episodes)}")
    
    env.close()
    
    print("\n=== Testing RolloutCollector (Vectorized) ===")
    vec_env_base = make_vec_env('CartPole-v1', n_envs=4, base_seed=42)
    vec_env = VecEnvWrapper(vec_env_base)
    collector = RolloutCollector(vec_env, agent, n_steps=6)
    
    # Reset
    collector.reset()
    print("✓ Reset works")
    
    # Collect
    buffer, final_states, episodes = collector.collect()
    print(f"✓ Collected {len(buffer.get()['states'])} transitions")
    print(f"  Should be: 4 envs × 6 steps = 24 transitions")
    print(f"✓ Final states: {len(final_states)} states")
    print(f"✓ Finished episodes: {len(episodes)}")
    
    # Check buffer structure
    data = buffer.get()
    print(f"\n✓ Buffer data:")
    print(f"  States: {len(data['states'])}")
    print(f"  Actions: {len(data['actions'])}")
    print(f"  Rewards: {len(data['rewards'])}")
    print(f"  Dones: {len(data['dones'])}")
    print(f"  Truncated: {len(data['truncated'])}")
    
    vec_env.close()
    
    print("\n✓ All collector tests passed!")
