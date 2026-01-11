"""
Comprehensive test suite for Person 2's modules
Run this to verify everything works before integration
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import shutil
import numpy as np


def test_metrics():
    """Test metrics module"""
    print("\n" + "="*60)
    print("TESTING: src/logging/metrics.py")
    print("="*60)
    
    from src.logging.metrics import (
        compute_mean_return,
        compute_std_return,
        compute_min_max_return,
        aggregate_seeds,
        compute_success_rate
    )
    
    # Test basic statistics
    returns = [100, 150, 120, 180, 90]
    
    mean_ret = compute_mean_return(returns)
    print(f"✓ Mean return: {mean_ret:.2f}")
    assert abs(mean_ret - 128.0) < 1, "Mean calculation error"
    
    std_ret = compute_std_return(returns)
    print(f"✓ Std return: {std_ret:.2f}")
    
    min_ret, max_ret = compute_min_max_return(returns)
    print(f"✓ Min/Max: {min_ret:.2f}, {max_ret:.2f}")
    assert min_ret == 90 and max_ret == 180, "Min/Max error"
    
    # Test success rate
    success_rate = compute_success_rate([500, 450, 500, 490], threshold=500)
    print(f"✓ Success rate: {success_rate:.2%}")
    assert success_rate == 0.5, "Success rate error"
    
    # Test aggregation
    seed1 = [100, 150, 200]
    seed2 = [90, 140, 210]
    seed3 = [110, 160, 190]
    
    mean, min_vals, max_vals = aggregate_seeds([seed1, seed2, seed3])
    print(f"✓ Aggregation works:")
    print(f"  Mean: {mean}")
    print(f"  Min: {min_vals}")
    print(f"  Max: {max_vals}")
    
    assert len(mean) == 3, "Aggregation length error"
    assert np.allclose(mean, [100, 150, 200]), "Aggregation mean error"
    
    print("✅ ALL METRICS TESTS PASSED!\n")
    return True


def test_vec_env():
    """Test vectorized environments"""
    print("\n" + "="*60)
    print("TESTING: src/rollout/vec_env.py")
    print("="*60)
    
    from src.rollout.vec_env import make_single_env, make_vec_env, VecEnvWrapper
    
    # Test single env creation
    env_fn = make_single_env('CartPole-v1', seed=42)
    env = env_fn()
    obs, _ = env.reset()
    print(f"✓ Single env obs shape: {obs.shape}")
    assert obs.shape == (4,), "Single env obs shape error"
    env.close()
    
    # Test vectorized env
    vec_env = make_vec_env('CartPole-v1', n_envs=4, base_seed=42)
    obs, _ = vec_env.reset()
    print(f"✓ Vec env obs shape: {obs.shape}")
    assert obs.shape == (4, 4), "Vec env obs shape error"
    
    # Test stepping
    actions = np.array([0, 1, 0, 1])
    obs, rewards, terminated, truncated, infos = vec_env.step(actions)
    print(f"✓ Step works - obs: {obs.shape}, rewards: {rewards.shape}")
    assert rewards.shape == (4,), "Rewards shape error"
    
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
    
    print("✅ ALL VEC_ENV TESTS PASSED!\n")
    return True


def test_collector():
    """Test rollout collectors"""
    print("\n" + "="*60)
    print("TESTING: src/rollout/collector.py")
    print("="*60)
    
    import gymnasium as gym
    from src.config import get_agent_config
    from src.algo.a2c_agent import A2CAgent
    from src.rollout.vec_env import make_vec_env, VecEnvWrapper
    from src.rollout.collector import RolloutCollector, SingleEnvCollector
    
    # Create agent
    config = get_agent_config(0, seed=42)
    agent = A2CAgent(state_dim=4, action_dim=2, config=config)
    
    print("\n--- Testing SingleEnvCollector ---")
    env = gym.make('CartPole-v1')
    collector = SingleEnvCollector(env, agent, n_steps=5)
    
    # Reset
    collector.reset()
    print("✓ Reset works")
    
    # Collect
    buffer, final_state, episodes = collector.collect()
    print(f"✓ Collected {len(buffer.get()['states'])} transitions")
    assert len(buffer.get()['states']) == 5, "Wrong number of transitions"
    print(f"✓ Final state shape: {final_state.shape}")
    assert final_state.shape == (4,), "Wrong final state shape"
    print(f"✓ Finished episodes: {len(episodes)}")
    
    env.close()
    
    print("\n--- Testing RolloutCollector (Vectorized) ---")
    vec_env_base = make_vec_env('CartPole-v1', n_envs=4, base_seed=42)
    vec_env = VecEnvWrapper(vec_env_base)
    collector = RolloutCollector(vec_env, agent, n_steps=6)
    
    # Reset
    collector.reset()
    print("✓ Reset works")
    
    # Collect
    buffer, final_states, episodes = collector.collect()
    n_transitions = len(buffer.get()['states'])
    print(f"✓ Collected {n_transitions} transitions")
    print(f"  Expected: 4 envs × 6 steps = 24 transitions")
    assert n_transitions == 24, f"Wrong number of transitions: {n_transitions}"
    
    print(f"✓ Final states: {len(final_states)} states")
    assert len(final_states) == 4, "Wrong number of final states"
    
    print(f"✓ Finished episodes: {len(episodes)}")
    
    # Check buffer structure
    data = buffer.get()
    print(f"\n✓ Buffer data:")
    print(f"  States: {len(data['states'])}")
    print(f"  Actions: {len(data['actions'])}")
    print(f"  Rewards: {len(data['rewards'])}")
    print(f"  Dones: {len(data['dones'])}")
    print(f"  Truncated: {len(data['truncated'])}")
    
    assert len(data['states']) == len(data['actions']) == len(data['rewards']), "Buffer size mismatch"
    
    vec_env.close()
    
    print("✅ ALL COLLECTOR TESTS PASSED!\n")
    return True


def test_evaluator():
    """Test evaluator"""
    print("\n" + "="*60)
    print("TESTING: src/eval/evaluator.py")
    print("="*60)
    
    from src.config import get_agent_config
    from src.algo.a2c_agent import A2CAgent
    from src.eval.evaluator import evaluate_agent, Evaluator
    
    # Create agent
    config = get_agent_config(0, seed=42)
    agent = A2CAgent(state_dim=4, action_dim=2, config=config)
    
    # Evaluate
    print("\n--- Evaluating untrained agent ---")
    stats = evaluate_agent(agent, n_episodes=5, record_trajectory=True)
    
    print(f"✓ Mean return: {stats['mean_return']:.2f}")
    print(f"✓ Std return: {stats['std_return']:.2f}")
    print(f"✓ Min/Max: {stats['min_return']:.0f} / {stats['max_return']:.0f}")
    print(f"✓ Returns: {stats['returns']}")
    
    assert len(stats['returns']) == 5, "Wrong number of episodes"
    assert 'mean_return' in stats, "Missing mean_return"
    
    if 'trajectory' in stats:
        traj = stats['trajectory']
        print(f"\n✓ Trajectory recorded:")
        print(f"  Length: {traj['length']}")
        print(f"  States shape: {traj['states'].shape}")
        print(f"  Values shape: {traj['values'].shape}")
        print(f"  First 5 values: {traj['values'][:5]}")
        
        assert traj['states'].shape[1] == 4, "Wrong state dimension"
    
    print("✅ ALL EVALUATOR TESTS PASSED!\n")
    return True


def test_value_trajectory():
    """Test value trajectory"""
    print("\n" + "="*60)
    print("TESTING: src/eval/value_traj.py")
    print("="*60)
    
    import gymnasium as gym
    from src.config import get_agent_config
    from src.algo.a2c_agent import A2CAgent
    from src.eval.value_traj import extract_value_trajectory, save_value_trajectory
    
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
    
    assert traj['length'] > 0, "Empty trajectory"
    assert len(traj['values']) == traj['length'], "Values length mismatch"
    
    # Save trajectory
    save_value_trajectory(traj, Path('test_outputs'), step=0)
    
    # Check file exists
    assert (Path('test_outputs/value_traj_step_0.csv')).exists(), "Trajectory file not created"
    print("✓ Trajectory saved successfully")
    
    # Cleanup
    env.close()
    shutil.rmtree('test_outputs')
    
    print("✅ ALL VALUE_TRAJ TESTS PASSED!\n")
    return True


def test_logger():
    """Test logger"""
    print("\n" + "="*60)
    print("TESTING: src/logging/logger.py")
    print("="*60)
    
    from src.logging.logger import Logger
    
    # Create test config
    config = {
        'agent_id': 0,
        'seed': 42,
        'gamma': 0.99,
        'actor_lr': 1e-5,
        'critic_lr': 1e-3
    }
    
    # Create logger
    logger = Logger(Path('test_outputs/agent0/seed0'), config)
    
    # Log training metrics
    print("\n--- Logging training metrics ---")
    for step in range(0, 5000, 1000):
        logger.log_train(
            step=step,
            episode_return=50.0 + step/100,
            actor_loss=0.1,
            critic_loss=0.05,
            entropy=0.6,
            mean_value=100.0,
            mean_advantage=0.0
        )
    
    print("✓ Training logs written")
    
    # Log evaluation metrics
    print("\n--- Logging evaluation metrics ---")
    for step in range(0, 60000, 20000):
        logger.log_eval(
            step=step,
            mean_return=200.0 + step/100,
            std_return=20.0,
            min_return=150.0,
            max_return=250.0,
            mean_length=200.0
        )
    
    print("✓ Evaluation logs written")
    
    # Check files exist
    assert (Path('test_outputs/agent0/seed0/config.json')).exists(), "Config not saved"
    assert (Path('test_outputs/agent0/seed0/train_log.csv')).exists(), "Train log not saved"
    assert (Path('test_outputs/agent0/seed0/eval_log.csv')).exists(), "Eval log not saved"
    
    print("✓ All files created")
    
    # Read and verify CSV
    import csv
    with open('test_outputs/agent0/seed0/train_log.csv', 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        print(f"✓ Train log has {len(rows)} rows")
        assert len(rows) == 5, "Wrong number of training log rows"
    
    with open('test_outputs/agent0/seed0/eval_log.csv', 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        print(f"✓ Eval log has {len(rows)} rows")
        assert len(rows) == 3, "Wrong number of eval log rows"
    
    # Cleanup
    shutil.rmtree('test_outputs')
    
    print("✅ ALL LOGGER TESTS PASSED!\n")
    return True


def test_integration():
    """Integration test: Collector + Evaluator + Logger"""
    print("\n" + "="*60)
    print("INTEGRATION TEST: Collector + Evaluator + Logger")
    print("="*60)
    
    import gymnasium as gym
    from src.config import get_agent_config
    from src.algo.a2c_agent import A2CAgent
    from src.rollout.vec_env import make_vec_env, VecEnvWrapper
    from src.rollout.collector import RolloutCollector
    from src.eval.evaluator import evaluate_agent
    from src.logging.logger import Logger
    
    # Setup
    config = get_agent_config(0, seed=42)
    agent = A2CAgent(state_dim=4, action_dim=2, config=config)
    
    # Create logger
    logger = Logger(Path('test_outputs/integration/seed0'), config.__dict__)
    
    # Create collector
    vec_env = VecEnvWrapper(make_vec_env('CartPole-v1', n_envs=2, base_seed=42))
    collector = RolloutCollector(vec_env, agent, n_steps=5)
    collector.reset()
    
    print("\n--- Simulating training loop ---")
    step = 0
    for iteration in range(5):
        # Collect data
        buffer, final_states, episodes = collector.collect()
        step += len(buffer.get()['states'])
        
        # Update agent
        metrics = agent.update(buffer, final_states)
        
        # Log finished episodes
        for ep in episodes:
            logger.log_train(
                step=step,
                episode_return=ep['return'],
                actor_loss=metrics['actor_loss'],
                critic_loss=metrics['critic_loss'],
                entropy=metrics['entropy'],
                mean_value=metrics['mean_value'],
                mean_advantage=metrics['mean_advantage']
            )
        
        print(f"  Iteration {iteration}: Step {step}, {len(episodes)} episodes finished")
    
    vec_env.close()
    
    # Evaluate
    print("\n--- Evaluating agent ---")
    eval_stats = evaluate_agent(agent, n_episodes=3, record_trajectory=True)
    
    logger.log_eval(
        step=step,
        mean_return=eval_stats['mean_return'],
        std_return=eval_stats['std_return'],
        min_return=eval_stats['min_return'],
        max_return=eval_stats['max_return'],
        mean_length=eval_stats['mean_length']
    )
    
    print(f"  Eval mean return: {eval_stats['mean_return']:.2f}")
    
    # Save value trajectory
    if 'trajectory' in eval_stats:
        logger.save_value_trajectory(eval_stats['trajectory'], step)
    
    print("\n✓ Integration test workflow complete")
    
    # Verify files
    assert (Path('test_outputs/integration/seed0/config.json')).exists()
    assert (Path('test_outputs/integration/seed0/train_log.csv')).exists()
    assert (Path('test_outputs/integration/seed0/eval_log.csv')).exists()
    
    # Cleanup
    shutil.rmtree('test_outputs')
    
    print("✅ INTEGRATION TEST PASSED!\n")
    return True


def main():
    """Run all tests"""
    print("\n" + "🧪"*30)
    print("PERSON 2 MODULE TEST SUITE")
    print("🧪"*30)
    
    tests = [
        ("Metrics", test_metrics),
        ("Vectorized Environments", test_vec_env),
        ("Collectors", test_collector),
        ("Evaluator", test_evaluator),
        ("Value Trajectory", test_value_trajectory),
        ("Logger", test_logger),
        ("Integration", test_integration),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ {name} FAILED with error:")
            print(f"   {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\n✅ Person 2's modules are ready for integration!")
        return True
    else:
        print("\n⚠️  Some tests failed. Please fix before integration.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)