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


def collect_rollout(agent, envs, n_steps, buffer, states, episode_rewards, start_step):
    """
    Collect rollout data from environments
    
    Args:
        agent: A2CAgent
        envs: Vectorized environments
        n_steps: Number of steps to collect
        buffer: RolloutBuffer to store data
        states: Current states for each env (None to reset)
        episode_rewards: Running episode returns per env (None to init)
        start_step: Global step count at rollout start
        
    Returns:
        next_states: Updated states for each env
        episode_rewards: Updated running returns
        episode_events: List of (step, episode_return) completed during rollout
    """
    num_envs = envs.num_envs if hasattr(envs, 'num_envs') else 1
    episode_events = []
    if states is None:
        states, _ = envs.reset()
    if episode_rewards is None:
        episode_rewards = [0.0] * num_envs
    
    for step in range(n_steps):
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
        dones = np.asarray(dones)
        truncated = np.asarray(truncated)
        rewards = np.asarray(rewards)
        
        # Store transitions
        for env_idx in range(num_envs):
            # Get actual reward (before masking)
            if isinstance(infos, dict) and 'true_reward' in infos:
                actual_reward = infos['true_reward'][env_idx]
            else:
                actual_reward = rewards[env_idx]
            
            episode_rewards[env_idx] += actual_reward

            # Use final observation for terminal/truncated steps (autoreset SAME_STEP)
            next_state_for_buffer = next_states[env_idx]
            if isinstance(infos, dict) and 'final_obs' in infos:
                if bool(infos.get('_final_obs', [False] * num_envs)[env_idx]):
                    next_state_for_buffer = infos['final_obs'][env_idx]
            
            # Store transition
            buffer.push(
                state=states[env_idx],
                action=actions[env_idx],
                reward=rewards[env_idx],  # Use masked reward for learning
                next_state=next_state_for_buffer,
                done=bool(dones[env_idx]),
                truncated=bool(truncated[env_idx]),
                log_prob=log_probs[env_idx],
                env_id=env_idx,
            )
            
            # Track completed episodes
            if dones[env_idx] or truncated[env_idx]:
                global_step = start_step + (step + 1) * num_envs
                episode_events.append((global_step, episode_rewards[env_idx]))
                episode_rewards[env_idx] = 0.0
        
        # Update states
        states = next_states
    
    return states, episode_rewards, episode_events


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
    states = None
    episode_rewards = None
    next_log_step = config.log_interval
    next_eval_step = config.eval_interval
    
    progress_bar = tqdm(total=config.max_steps, desc=f"Agent {agent_id}")
    
    while total_steps < config.max_steps:
        # Collect rollout
        buffer = RolloutBuffer()
        states, episode_rewards, episode_events = collect_rollout(
            agent, envs, config.n_steps, buffer, states, episode_rewards, total_steps
        )
        
        # Update agent
        metrics = agent.update(buffer, list(states))
        
        # Update counters
        steps_collected = config.num_workers * config.n_steps
        total_steps += steps_collected
        episode_count += len(episode_events)

        # Log episodic returns as soon as episodes finish
        for step, ep_return in episode_events:
            logger.log_episode(step, ep_return)
        
        # Log training metrics
        while total_steps >= next_log_step:
            log_data = {
                'step': next_log_step,
                'actor_loss': metrics['actor_loss'],
                'critic_loss': metrics['critic_loss'],
                'entropy': metrics['entropy'],
                'mean_value': metrics['mean_value'],
            }
            logger.log_train(next_log_step, log_data)
            next_log_step += config.log_interval
        
        # Evaluation
        while total_steps >= next_eval_step:
            from src.eval import evaluate
            from src.eval.value_traj import extract_value_trajectory, save_value_trajectory
            from src.envs import make_env
            
            eval_env = make_env(seed=seed, reward_mask_prob=0.0)
            eval_metrics = evaluate(agent, eval_env, num_episodes=config.eval_episodes)
            
            # Save value trajectory for this evaluation step
            traj = extract_value_trajectory(agent, eval_env)
            save_value_trajectory(traj, run_dir, step=next_eval_step)
            eval_env.close()
            
            logger.log_eval(next_eval_step, eval_metrics)
            
            progress_bar.set_postfix({
                'eval_return': f"{eval_metrics['mean_return']:.1f}",
                'episodes': episode_count
            })
            next_eval_step += config.eval_interval
        
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
