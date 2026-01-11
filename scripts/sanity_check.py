"""
Sanity check script to verify Phase 1 is working correctly
Run this before moving to Phase 2
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import set_seed, get_device, to_tensor, to_numpy
from src.utils import save_json, load_json, ensure_dir
from src.config import get_agent_config
from src.envs import make_env, make_vec_env
import numpy as np
import torch


def test_utils():
    """Test utility functions"""
    print("=" * 60)
    print("TESTING UTILITIES")
    print("=" * 60)
    
    # Test seeding
    set_seed(42)
    r1 = np.random.rand()
    set_seed(42)
    r2 = np.random.rand()
    assert r1 == r2, "Seeding not working!"
    print("✓ Seeding works")
    
    # Test device
    device = get_device()
    print(f"✓ Device: {device}")
    
    # Test tensor conversion
    x_np = np.array([1.0, 2.0, 3.0])
    x_torch = to_tensor(x_np, device)
    x_back = to_numpy(x_torch)
    assert np.allclose(x_np, x_back), "Tensor conversion failed!"
    print("✓ Tensor conversion works")
    
    # Test I/O
    test_dir = ensure_dir("test_outputs")
    test_data = {'a': 1, 'b': 2}
    save_json(test_data, "test_outputs/test.json")
    loaded_data = load_json("test_outputs/test.json")
    assert loaded_data == test_data, "JSON I/O failed!"
    print("✓ JSON I/O works")
    
    # Cleanup
    import shutil
    shutil.rmtree("test_outputs")
    
    print()


def test_config():
    """Test configuration"""
    print("=" * 60)
    print("TESTING CONFIGURATION")
    print("=" * 60)
    
    for agent_id in range(5):
        config = get_agent_config(agent_id, seed=42)
        print(f"✓ Agent {agent_id}: K={config.num_workers}, n={config.n_steps}, "
              f"mask_prob={config.reward_mask_prob:.1f}")
    
    print()


def test_environments():
    """Test environment creation"""
    print("=" * 60)
    print("TESTING ENVIRONMENTS")
    print("=" * 60)
    
    # Test single environment
    env = make_env(seed=42, reward_mask_prob=0.0)
    state, info = env.reset()
    assert state.shape == (4,), f"Wrong state shape: {state.shape}"
    print(f"✓ Single env state shape: {state.shape}")
    
    action = env.action_space.sample()
    next_state, reward, done, truncated, info = env.step(action)
    assert 'should_bootstrap' in info, "Missing bootstrapping info!"
    print(f"✓ Single env step works, info keys: {list(info.keys())}")
    
    env.close()
    
    # Test vectorized environment
    num_envs = 4
    envs = make_vec_env(num_envs=num_envs, seed=42, reward_mask_prob=0.0)
    states, infos = envs.reset()
    assert states.shape == (num_envs, 4), f"Wrong states shape: {states.shape}"
    print(f"✓ Vectorized env states shape: {states.shape}")
    
    # FIX: Sample actions correctly for vectorized env
    actions = envs.action_space.sample()  # This returns the correct shape
    print(f"✓ Actions shape: {actions.shape} (type: {type(actions)})")
    
    next_states, rewards, dones, truncated, infos = envs.step(actions)
    assert rewards.shape == (num_envs,), f"Wrong rewards shape: {rewards.shape}"
    print(f"✓ Vectorized env step works")
    
    envs.close()
    
    # Test stochastic rewards
    env = make_env(seed=42, reward_mask_prob=0.9)
    state, info = env.reset()
    
    masked_count = 0
    total_steps = 100
    for _ in range(total_steps):
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        if 'reward_was_masked' in info and info['reward_was_masked']:
            masked_count += 1
        if done or truncated:
            state, info = env.reset()
    
    mask_rate = masked_count / total_steps
    print(f"✓ Stochastic rewards: {mask_rate:.1%} masked (expected ~90%)")
    
    env.close()
    
    print()


def test_bootstrapping():
    """Test bootstrapping flags"""
    print("=" * 60)
    print("TESTING BOOTSTRAPPING")
    print("=" * 60)
    
    env = make_env(seed=42)
    state, info = env.reset()
    
    # Run until episode ends
    step_count = 0
    while True:
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        step_count += 1
        
        if done or truncated:
            print(f"Episode ended at step {step_count}")
            print(f"  Terminal: {info['is_terminal']}")
            print(f"  Truncated: {info['is_truncated']}")
            print(f"  Should bootstrap: {info['should_bootstrap']}")
            
            # Verify logic
            if info['is_terminal']:
                assert not info['should_bootstrap'], "Terminal states should not bootstrap!"
                print("  ✓ Terminal state logic correct")
            if info['is_truncated']:
                assert info['should_bootstrap'], "Truncated states should bootstrap!"
                print("  ✓ Truncated state logic correct")
            
            break
    
    env.close()
    print()


def main():
    """Run all sanity checks"""
    print("\n" + "=" * 60)
    print("PHASE 1 SANITY CHECK")
    print("=" * 60)
    print()
    
    try:
        test_utils()
        test_config()
        test_environments()
        test_bootstrapping()
        
        print("=" * 60)
        print("✓ ALL PHASE 1 CHECKS PASSED!")
        print("=" * 60)
        print()
        print("You are ready to move to Phase 2 (Neural Networks)")
        print()
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ SANITY CHECK FAILED!")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)