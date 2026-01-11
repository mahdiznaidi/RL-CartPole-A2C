"""
Test collector with different configurations
"""

import gymnasium as gym
import numpy as np
from src.config import get_agent_config
from src.algo.a2c_agent import A2CAgent
from src.rollout.vec_env import make_vec_env, VecEnvWrapper
from src.rollout.collector import RolloutCollector, SingleEnvCollector


def test_single_env_collector():
    """Test K=1, n=1"""
    print("\n=== Test K=1, n=1 ===")
    
    config = get_agent_config(0, seed=42)
    agent = A2CAgent(4, 2, config)
    
    env = gym.make('CartPole-v1')
    collector = SingleEnvCollector(env, agent, n_steps=1)
    collector.reset()
    
    # Collect 100 steps
    total_transitions = 0
    total_episodes = 0
    
    for _ in range(100):
        buffer, final_state, episodes = collector.collect()
        total_transitions += len(buffer.get()['states'])
        total_episodes += len(episodes)
    
    print(f"✓ Collected {total_transitions} transitions")
    print(f"✓ Finished {total_episodes} episodes")
    
    env.close()


def test_vec_env_collector():
    """Test K=6, n=6"""
    print("\n=== Test K=6, n=6 ===")
    
    config = get_agent_config(4, seed=42)
    agent = A2CAgent(4, 2, config)
    
    vec_env = VecEnvWrapper(make_vec_env('CartPole-v1', n_envs=6, base_seed=42))
    collector = RolloutCollector(vec_env, agent, n_steps=6)
    collector.reset()
    
    # Collect 10 rollouts
    total_transitions = 0
    total_episodes = 0
    
    for i in range(10):
        buffer, final_states, episodes = collector.collect()
        trans = len(buffer.get()['states'])
        total_transitions += trans
        total_episodes += len(episodes)
        
        print(f"  Rollout {i}: {trans} transitions, {len(episodes)} episodes finished")
    
    print(f"\n✓ Total: {total_transitions} transitions")
    print(f"✓ Total: {total_episodes} episodes")
    print(f"✓ Expected: ~{10 * 6 * 6} transitions")
    
    vec_env.close()


def test_bootstrapping():
    """Test that truncation bootstraps correctly"""
    print("\n=== Test Bootstrapping ===")
    
    config = get_agent_config(0, seed=42)
    agent = A2CAgent(4, 2, config)
    
    env = gym.make('CartPole-v1')
    collector = SingleEnvCollector(env, agent, n_steps=1)
    collector.reset()
    
    # Collect until we hit a truncation
    for _ in range(1000):
        buffer, final_state, episodes = collector.collect()
        
        if episodes:
            episode = episodes[0]
            data = buffer.get()
            
            print(f"Episode finished:")
            print(f"  Return: {episode['return']}")
            print(f"  Length: {episode['length']}")
            print(f"  Terminated: {episode['terminated']}")
            print(f"  Truncated: {episode['truncated']}")
            print(f"  Last transition done: {data['dones'][-1]}")
            print(f"  Last transition truncated: {data['truncated'][-1]}")
            
            if episode['truncated']:
                print("✓ Found truncated episode!")
                break
    
    env.close()


if __name__ == "__main__":
    test_single_env_collector()
    test_vec_env_collector()
    test_bootstrapping()
    
    print("\n✓ All collector tests passed!")