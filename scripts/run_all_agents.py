# scripts/run_all_agents.py
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed training for multiple agents.")
    parser.add_argument(
        "--agents",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Agent IDs to run (default: 1 2 3 4)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds to run per agent (default: 0 1 2)",
    )
    return parser.parse_args()


def main():
    project_root = Path(__file__).resolve().parents[1]
    python_exe = sys.executable

    args = _parse_args()
    agents = args.agents
    seeds = args.seeds

    for agent in agents:
        for seed in seeds:
            cmd = [
                python_exe,
                str(project_root / "scripts" / "train.py"),
                "--agent", str(agent),
                "--seed", str(seed),
            ]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)

    print("[OK] All training runs completed.")


if __name__ == "__main__":
    main()
