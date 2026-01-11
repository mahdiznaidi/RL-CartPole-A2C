# scripts/dev_fake_run.py
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    root = Path(__file__).resolve().parents[1]
    run = root / "outputs" / "runs" / "agent1" / "seed0"
    run.mkdir(parents=True, exist_ok=True)

    steps = np.arange(0, 200_000, 1000)
    train = pd.DataFrame({
        "step": steps,
        "episode_return": np.clip(np.linspace(20, 450, len(steps)) + np.random.randn(len(steps))*10, 0, 500),
        "actor_loss": np.abs(np.random.randn(len(steps))*0.1),
        "critic_loss": np.abs(np.random.randn(len(steps))*0.05),
    })
    train.to_csv(run / "train_log.csv", index=False)

    eval_steps = np.arange(20_000, 200_001, 20_000)
    eval_df = pd.DataFrame({
        "step": eval_steps,
        "eval_return_mean": np.clip(np.linspace(50, 500, len(eval_steps)) + np.random.randn(len(eval_steps))*15, 0, 500),
    })
    eval_df.to_csv(run / "eval_log.csv", index=False)

    print("Fake run created:", run)

if __name__ == "__main__":
    main()
