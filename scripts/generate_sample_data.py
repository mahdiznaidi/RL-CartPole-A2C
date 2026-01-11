"""
Generate sample data for Person 3 to test plotting
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.logging.logger import Logger

def generate_sample_data():
    """Generate realistic sample data for testing plots"""
    
    print("Generating sample data for Person 3...")
    
    # Simulate 3 seeds for Agent 0
    for seed in [0, 1, 2]:
        config = {
            'agent_id': 0,
            'seed': seed,
            'K': 1,
            'n_steps': 1,
            'gamma': 0.99,
            'actor_lr': 1e-5,
            'critic_lr': 1e-3
        }
        
        logger = Logger(Path(f'outputs/runs/agent0/seed{seed}'), config)
        
        # Simulate training with improving returns
        np.random.seed(seed)
        step = 0
        episode_return = 20.0
        
        while step < 100000:
            # Simulate episode
            episode_return += np.random.randn() * 5 + 0.1  # Gradually improve
            episode_return = np.clip(episode_return, 10, 500)
            
            step += int(np.random.randint(10, 50))
            
            # Log training
            logger.log_train(
                step=step,
                episode_return=episode_return,
                actor_loss=0.1 * np.exp(-step/50000),
                critic_loss=0.05 * np.exp(-step/50000),
                entropy=0.6 * np.exp(-step/100000),
                mean_value=episode_return * 0.99,
                mean_advantage=np.random.randn() * 0.1
            )
            
            # Log evaluation every 20k
            if step % 20000 < 50:
                eval_return = episode_return + np.random.randn() * 10
                logger.log_eval(
                    step=step,
                    mean_return=eval_return,
                    std_return=abs(np.random.randn() * 20),
                    min_return=eval_return - 30,
                    max_return=min(500, eval_return + 30),
                    mean_length=eval_return
                )
        
        print(f"✓ Generated data for Agent 0, seed {seed}")
    
    print("\n✅ Sample data generated at: outputs/runs/agent0/")
    print("\nPerson 3 can now test plotting with:")
    print("  python scripts/test_plotting.py")

if __name__ == "__main__":
    generate_sample_data()