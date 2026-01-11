# Person 2 Interface Documentation

**Status**: ✅ All modules tested and ready for integration

**Last Updated**: January 11, 2026

---

## 📦 Completed Modules

### 1. Data Collection System
- ✅ `src/rollout/vec_env.py` - Vectorized environments (K workers)
- ✅ `src/rollout/collector.py` - Rollout collection (n-step)
  - `SingleEnvCollector` for K=1
  - `RolloutCollector` for K>1

### 2. Evaluation System
- ✅ `src/eval/evaluator.py` - Agent evaluation (greedy policy)
- ✅ `src/eval/value_traj.py` - Value trajectory extraction

### 3. Logging System
- ✅ `src/logging/logger.py` - Training/eval logging
- ✅ `src/logging/metrics.py` - Metric computation and aggregation

---

## 🔌 For Person 1: Training Script Integration

### Quick Start Example
```python
from pathlib import Path
import gymnasium as gym
from src.config import get_agent_config
from src.algo.a2c_agent import A2CAgent
from src.rollout.collector import SingleEnvCollector, RolloutCollector
from src.rollout.vec_env import make_vec_env, VecEnvWrapper
from src.eval.evaluator import evaluate_agent
from src.logging.logger import Logger

# Setup
config = get_agent_config(agent_id=0, seed=42)
agent = A2CAgent(state_dim=4, action_dim=2, config=config)
logger = Logger(Path(f'outputs/runs/agent{agent_id}/seed{seed}'), config.__dict__)

# Choose collector based on K
if config.K == 1:
    env = gym.make('CartPole-v1')
    collector = SingleEnvCollector(env, agent, n_steps=config.n_steps)
else:
    vec_env = VecEnvWrapper(make_vec_env('CartPole-v1', n_envs=config.K, base_seed=seed))
    collector = RolloutCollector(vec_env, agent, n_steps=config.n_steps)

collector.reset()

# Training loop
step = 0
while step < 500000:
    # Collect rollout
    if config.K == 1:
        buffer, final_state, episodes = collector.collect()
        final_states = [final_state]  # Wrap in list for update()
    else:
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
    
    # Evaluate every 20k steps
    if step % 20000 == 0:
        eval_stats = evaluate_agent(
            agent, 
            n_episodes=10, 
            seed=seed+1000,
            record_trajectory=True
        )
        
        logger.log_eval(
            step=step,
            mean_return=eval_stats['mean_return'],
            std_return=eval_stats['std_return'],
            min_return=eval_stats['min_return'],
            max_return=eval_stats['max_return'],
            mean_length=eval_stats['mean_length']
        )
        
        logger.save_value_trajectory(eval_stats['trajectory'], step)
```

### Agent Configuration Table

| Agent | K (workers) | n (steps) | Collector to Use | Notes |
|-------|-------------|-----------|------------------|-------|
| 0 | 1 | 1 | `SingleEnvCollector` | Basic A2C |
| 1 | 1 | 1 | `SingleEnvCollector` | + stochastic rewards |
| 2 | 6 | 1 | `RolloutCollector` | Parallel workers |
| 3 | 1 | 6 | `SingleEnvCollector` | N-step returns |
| 4 | 6 | 6 | `RolloutCollector` | Both combined |

### Important Notes

1. **Step Counting**: Each transition = 1 step. For K=6, n=6, one rollout = 36 steps.

2. **Final States**: 
   - `SingleEnvCollector` returns a **single array**, wrap in list: `[final_state]`
   - `RolloutCollector` returns a **list of arrays**, use directly

3. **Episode Logging**: Episodes finish at random times. Only log when `episodes` list is not empty.

4. **Evaluation Seed**: Use `seed + 1000` for evaluation to avoid correlation with training.

---

## 📊 For Person 3: Data Format and Plotting

### Directory Structure
```
outputs/runs/
├── agent0/
│   ├── seed0/
│   │   ├── config.json
│   │   ├── train_log.csv
│   │   ├── eval_log.csv
│   │   ├── value_traj_step_20000.csv
│   │   ├── value_traj_step_40000.csv
│   │   └── ...
│   ├── seed1/
│   └── seed2/
├── agent1/
│   └── ...
└── agent4/
```

### CSV Formats

**`train_log.csv`** (logged whenever episode finishes):
```csv
step,episode_return,actor_loss,critic_loss,entropy,mean_value,mean_advantage
1234,50.0,0.123,0.045,0.678,120.5,0.012
2456,65.0,0.098,0.038,0.654,135.2,0.008
```

**`eval_log.csv`** (logged every 20k steps):
```csv
step,mean_return,std_return,min_return,max_return,mean_length
20000,450.0,25.5,400.0,490.0,450.0
40000,480.0,18.2,450.0,500.0,480.0
```

**`value_traj_step_XXXXX.csv`**:
```csv
step_in_episode,value,reward,action,state_0,state_1,state_2,state_3
0,245.3,1.0,0,0.01,-0.02,0.05,0.03
1,244.5,1.0,1,0.02,-0.01,0.04,0.02
```

### Loading and Aggregating Data
```python
from pathlib import Path
from src.logging.metrics import aggregate_across_seeds
import matplotlib.pyplot as plt

# Load and aggregate training returns across 3 seeds
mean, min_vals, max_vals = aggregate_across_seeds(
    agent_dir=Path('outputs/runs/agent0'),
    seeds=[0, 1, 2],
    log_type='train',
    metric_name='episode_return'
)

# Plot with error bands
plt.figure(figsize=(10, 6))
x = range(len(mean))
plt.plot(x, mean, label='Agent 0', linewidth=2)
plt.fill_between(x, min_vals, max_vals, alpha=0.3)
plt.xlabel('Data Point')
plt.ylabel('Episode Return')
plt.title('Training Returns - Agent 0')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('outputs/figures/agent0_train_return.png', dpi=150)
```

### Handling Different Data Lengths

Seeds may have different numbers of logged episodes. The `aggregate_across_seeds` function **automatically pads** shorter sequences with their last value.

### Required Plots (from PDF)

1. **Training curves** - Episode returns over training steps
2. **Evaluation curves** - Mean returns every 20k steps  
3. **Loss curves** - Actor and critic losses
4. **Value trajectories** - V(s) along sample episodes
5. **Comparison plots** - All 5 agents on same plot

---

## 🧪 Testing

All modules have been tested:
```bash
# Run comprehensive test suite
python scripts/test_person2_modules.py

# Result: 7/7 tests passed ✅
```

---

## ❓ FAQ

### For Person 1

**Q: How do I handle the step counter?**  
A: Increment by the number of transitions collected:
```python
step += len(buffer.get()['states'])
```

**Q: What if an episode doesn't finish during a rollout?**  
A: That's fine! The `episodes` list will be empty. Only log when it's not empty.

**Q: Should I save model checkpoints?**  
A: Yes, save in the same directory: `logger.run_dir / 'checkpoint_XXXXX.pt'`

### For Person 3

**Q: What if some seeds are missing data?**  
A: The aggregation function will skip missing seeds and warn you.

**Q: How do I align data from different episodes?**  
A: Use the `step` column - it's the same across all seeds.

**Q: Should I smooth the curves?**  
A: No, show raw data. The min/max bands already show variance.

---

## 📞 Contact Person 2

If you have questions or issues, ask Person 2 about:
- Collector usage
- Data format
- Missing features
- Bug fixes

---

**Version**: 1.0  
**Status**: Ready for integration ✅