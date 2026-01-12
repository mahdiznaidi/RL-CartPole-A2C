# src/logging/logger.py
from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict


class CSVLogger:
    """
    Minimal CSV logger that writes:
    - config.json
    - train_log.csv
    - eval_log.csv
    - episode_log.csv
    """

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.train_csv = self.run_dir / "train_log.csv"
        self.eval_csv = self.run_dir / "eval_log.csv"
        self.episode_csv = self.run_dir / "episode_log.csv"

        self._train_header_written = self.train_csv.exists() and self.train_csv.stat().st_size > 0
        self._eval_header_written = self.eval_csv.exists() and self.eval_csv.stat().st_size > 0
        self._episode_header_written = self.episode_csv.exists() and self.episode_csv.stat().st_size > 0

    def save_config(self, config: Any) -> None:
        cfg_path = self.run_dir / "config.json"
        if is_dataclass(config):
            payload = asdict(config)
        elif isinstance(config, dict):
            payload = config
        else:
            payload = getattr(config, "__dict__", {"config": str(config)})

        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def log_train(self, row: Dict[str, Any]) -> None:
        self._append_row(self.train_csv, row, is_train=True)

    def log_eval(self, row: Dict[str, Any]) -> None:
        self._append_row(self.eval_csv, row, is_train=False)

    def log_episode(self, row: Dict[str, Any]) -> None:
        self._append_row(self.episode_csv, row, is_train=False, is_episode=True)

    def save_value_trajectory(self, step: int, traj: Dict[str, Any], filename_prefix: str = "value_traj") -> Path:
        out = self.run_dir / f"{filename_prefix}_eval_{step}.csv"
        keys = list(traj.keys())
        n = len(traj[keys[0]]) if keys else 0

        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for i in range(n):
                writer.writerow({k: traj[k][i] for k in keys})
        return out

    def _append_row(self, path: Path, row: Dict[str, Any], is_train: bool, is_episode: bool = False) -> None:
        if is_episode:
            header_written = self._episode_header_written
        else:
            header_written = self._train_header_written if is_train else self._eval_header_written

        if header_written:
            with path.open("r", encoding="utf-8") as f:
                header = f.readline().strip().split(",")
            row_to_write = {k: row.get(k, "") for k in header}
            with path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writerow(row_to_write)
            return

        header = list(row.keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerow(row)

        if is_train:
            self._train_header_written = True
        elif is_episode:
            self._episode_header_written = True
        else:
            self._eval_header_written = True


def get_run_dir(project_root: Path, agent_id: str, seed: int) -> Path:
    return project_root / "outputs" / "runs" / agent_id / f"seed{seed}"


class Logger:
    """
    Compatibility wrapper expected by training code.
    """

    def __init__(self, run_dir: Path, config: Any):
        self._csv = CSVLogger(run_dir)
        self._csv.save_config(config)

    def log_train(self, step: int, metrics: Dict[str, Any]) -> None:
        row = dict(metrics)
        row.setdefault("step", step)
        self._csv.log_train(row)

    def log_eval(self, step: int, metrics: Dict[str, Any]) -> None:
        row = dict(metrics)
        row.setdefault("step", step)
        self._csv.log_eval(row)

    def log_episode(self, step: int, episode_return: float) -> None:
        self._csv.log_episode({"step": step, "episode_return": episode_return})
