# src/logging/logger.py
from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional


class CSVLogger:
    """
    Logger minimaliste:
    - crée un dossier run_dir = outputs/runs/<agent_id>/seed<seed>/
    - écrit config.json
    - append train_log.csv et eval_log.csv
    """

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.train_csv = self.run_dir / "train_log.csv"
        self.eval_csv = self.run_dir / "eval_log.csv"

        self._train_header_written = self.train_csv.exists() and self.train_csv.stat().st_size > 0
        self._eval_header_written = self.eval_csv.exists() and self.eval_csv.stat().st_size > 0

    def save_config(self, config: Any) -> None:
        cfg_path = self.run_dir / "config.json"
        if is_dataclass(config):
            payload = asdict(config)
        elif isinstance(config, dict):
            payload = config
        else:
            # fallback: try __dict__
            payload = getattr(config, "__dict__", {"config": str(config)})

        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def log_train(self, row: Dict[str, Any]) -> None:
        """
        row exemple:
        {
          "step": 12000,
          "episode_return": 210.0,
          "actor_loss": 0.12,
          "critic_loss": 0.03,
          "entropy": 0.68
        }
        """
        self._append_row(self.train_csv, row, is_train=True)

    def log_eval(self, row: Dict[str, Any]) -> None:
        """
        row exemple:
        { "step": 20000, "eval_return_mean": 300.0 }
        """
        self._append_row(self.eval_csv, row, is_train=False)

    def save_value_trajectory(self, step: int, traj: Dict[str, Any], filename_prefix: str = "value_traj") -> Path:
        """
        Sauvegarde une trajectoire pour V(s) lors d'une éval.
        traj attendu:
          {
            "t": [0,1,2,...],
            "v": [..],
            "obs_0": [...], "obs_1": [...], ...
          }
        """
        out = self.run_dir / f"{filename_prefix}_eval_{step}.csv"
        keys = list(traj.keys())
        # longueur = len(traj[keys[0]])
        n = len(traj[keys[0]]) if keys else 0

        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for i in range(n):
                writer.writerow({k: traj[k][i] for k in keys})
        return out

    def _append_row(self, path: Path, row: Dict[str, Any], is_train: bool) -> None:
        # Ecrire header dynamiquement si nouveau fichier
        header_written = self._train_header_written if is_train else self._eval_header_written

        # Si on a déjà écrit un header, on conserve l'ordre existant
        if header_written:
            # Lire header existant
            with path.open("r", encoding="utf-8") as f:
                header = f.readline().strip().split(",")
            # Ajouter colonnes manquantes (rare, mais possible)
            # -> Simple stratégie: réécrire un nouveau fichier "wide" serait plus complexe.
            # On choisit plutôt: n'écrire que les colonnes existantes + ignorer nouvelles.
            row_to_write = {k: row.get(k, "") for k in header}
            with path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writerow(row_to_write)
            return

        # Sinon, écrire header basé sur row
        header = list(row.keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerow(row)

        if is_train:
            self._train_header_written = True
        else:
            self._eval_header_written = True


def get_run_dir(project_root: Path, agent_id: str, seed: int) -> Path:
    return project_root / "outputs" / "runs" / agent_id / f"seed{seed}"
