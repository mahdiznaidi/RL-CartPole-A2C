# scripts/run_all_agents.py
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[1]
    python_exe = sys.executable

    agents = ["agent1", "agent2", "agent3", "agent4"]
    seeds = [0, 1, 2]

    for agent in agents:
        for seed in seeds:
            cmd = [
                python_exe,
                str(project_root / "scripts" / "train.py"),
                "--agent", agent,
                "--seed", str(seed),
            ]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)

    print("[OK] All training runs completed.")


if __name__ == "__main__":
    main()
