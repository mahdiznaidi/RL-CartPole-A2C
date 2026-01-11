"""
Training script for A2C agents
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import torch
from tqdm import tqdm

from src.config import get_agent_config
from src.utils import set_seed, ensure_dir
from src.envs import make_vec_env
from src.algo import A2CAgent, RolloutBuffer
from src.logging import Logger


def collect_rollout(agent, envs, n_steps, buffer):
    """
    Collect rollout data from environments
    
    Args:
        agent: A2CAgent
        envs: Vectorized environments
        n_steps: Number of steps to collect
        buffer: RolloutBuffer to store data
        
    Returns:
        episode_returns: List of episode returns that completed during rollout
    """
    num_envs = envs.num_envs if hasattr(envs, 'num_envs') else 1
    episode_returns = []
    episode_rewards = [0.0] * num_envs
    
    for step in range(n_steps):
        # Get current states
        if step == 0:
            states, _ = envs.reset()
        
        # Select actions for all environments
        actions = []
        log_probs = []
        
        for env_idx in range(num_envs):
            state = states[env_idx]
            action, log_prob = agent.select_action(state, greedy=False)
            actions.append(action)
            log_probs.append(log_prob)
        
        actions = np.array(actions)
        
        # Step environments
        next_states, rewards, dones, truncated, infos = envs.step(actions)
        
        # Store transitions
        for env_idx in range(num_envs):
            # Get actual reward (before masking)
            if 'true_reward' in infos:
                actual_reward = infos['true_reward'][env_idx]
            else:
                actual_reward = rewards[env_idx]
            
            episode_rewards[env_idx] += actual_reward
            
            # Store transition
            buffer.push(
                state=states[env_idx],
                action=actions[env_idx],
                reward=rewards[env_idx],  # Use masked reward for learning
                next_state=next_states[env_idx],
                done=dones[env_idx],
                truncated=truncated[env_idx],
                log_prob=log_probs[env_idx]
            )
            
            # Track completed episodes
            if dones[env_idx] or truncated[env_idx]:
                episode_returns.append(episode_rewards[env_idx])
                episode_rewards[env_idx] = 0.0
        
        # Update states
        states = next_states
    
    return episode_returns


def train_agent(agent_id: int, seed: int):
    """
    Train a single agent
    
    Args:
        agent_id: Agent number (0-4)
        seed: Random seed
    """
    # Get configuration
    config = get_agent_config(agent_id, seed)
    
    print("=" * 60)
    print(f"TRAINING AGENT {agent_id} (SEED {seed})")
    print("=" * 60)
    print(f"K (workers): {config.num_workers}")
    print(f"n (steps): {config.n_steps}")
    print(f"Reward masking: {config.reward_mask_prob}")
    print(f"Max steps: {config.max_steps:,}")
    print("=" * 60)
    
    # Set seed
    set_seed(seed)
    
    # Create output directory
    run_dir = config.get_run_dir()
    ensure_dir(run_dir)
    
    # Create logger
    from src.logging import Logger
    logger = Logger(run_dir, config)
    
    # Create environments
    envs = make_vec_env(
        num_envs=config.num_workers,
        seed=seed,
        reward_mask_prob=config.reward_mask_prob
    )
    
    # Create agent
    agent = A2CAgent(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        config=config
    )
    
    # Training loop
    total_steps = 0
    episode_count = 0
    
    progress_bar = tqdm(total=config.max_steps, desc=f"Agent {agent_id}")
    
    while total_steps < config.max_steps:
        # Collect rollout
        buffer = RolloutBuffer()
        episode_returns = collect_rollout(
            agent, envs, config.n_steps, buffer
        )
        
        # Update agent
        final_next_states = [np.random.randn(config.state_dim)]  # Placeholder
        metrics = agent.update(buffer, final_next_states)
        
        # Update counters
        steps_collected = config.num_workers * config.n_steps
        total_steps += steps_collected
        episode_count += len(episode_returns)
        
        # Log training metrics
        if total_steps % config.log_interval == 0:
            log_data = {
                'step': total_steps,
                'actor_loss': metrics['actor_loss'],
                'critic_loss': metrics['critic_loss'],
                'entropy': metrics['entropy'],
                'mean_value': metrics['mean_value'],
            }
            
            if episode_returns:
                log_data['episode_return'] = np.mean(episode_returns)
            
            logger.log_train(total_steps, log_data)
        
        # Evaluation
        if total_steps % config.eval_interval == 0:
            from src.eval import evaluate
            from src.envs import make_env
            
            eval_env = make_env(seed=seed, reward_mask_prob=0.0)
            eval_metrics = evaluate(agent, eval_env, num_episodes=config.eval_episodes)
            eval_env.close()
            
            logger.log_eval(total_steps, eval_metrics)
            
            progress_bar.set_postfix({
                'eval_return': f"{eval_metrics['mean_return']:.1f}",
                'episodes': episode_count
            })
        
        progress_bar.update(steps_collected)
    
    progress_bar.close()
    envs.close()
    
    print(f"\n✓ Training complete for Agent {agent_id}, Seed {seed}")
    print(f"  Total episodes: {episode_count}")
    print(f"  Output: {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train A2C agent")
    parser.add_argument("--agent", type=int, required=True, help="Agent ID (0-4)")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    
    args = parser.parse_args()
    
    train_agent(args.agent, args.seed)


if __name__ == "__main__":
    main()