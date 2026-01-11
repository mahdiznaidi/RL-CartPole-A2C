<<<<<<< HEAD
"""
Training and evaluation logging
"""

import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from src.utils.io import save_json, ensure_dir
=======
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
>>>>>>> ef9b537a0fcad5f8b0cb77374192c2375ee31de1


class Logger:
    """
<<<<<<< HEAD
    Logger for training and evaluation metrics
    
    Saves:
    - config.json: Configuration
    - train_log.csv: Training metrics (every step/batch)
    - eval_log.csv: Evaluation metrics (every 20k steps)
    - value_traj_step_X.csv: Value trajectories
    
    Args:
        run_dir: Directory to save logs (e.g., outputs/runs/agent0/seed0)
        config: Configuration dictionary to save
        
    Example:
        >>> logger = Logger(Path('outputs/runs/agent0/seed0'), config_dict)
        >>> logger.log_train(step=1000, episode_return=50.0, actor_loss=0.1)
        >>> logger.log_eval(step=20000, mean_return=450.0)
    """
    
    def __init__(self, run_dir: Path, config: Dict[str, Any]):
        self.run_dir = Path(run_dir)
        ensure_dir(self.run_dir)
        
        # Save config
        save_json(config, self.run_dir / 'config.json')
        
        # CSV file paths
        self.train_log_path = self.run_dir / 'train_log.csv'
        self.eval_log_path = self.run_dir / 'eval_log.csv'
        
        # Training log fields
        self.train_fields = [
            'step',
            'episode_return',  # From finished episodes
            'actor_loss',
            'critic_loss',
            'entropy',
            'mean_value',
            'mean_advantage'
        ]
        
        # Evaluation log fields
        self.eval_fields = [
            'step',
            'mean_return',
            'std_return',
            'min_return',
            'max_return',
            'mean_length'
        ]
        
        # Initialize CSV files with headers
        self._init_csv(self.train_log_path, self.train_fields)
        self._init_csv(self.eval_log_path, self.eval_fields)
        
        print(f"✓ Logger initialized at {self.run_dir}")
    
    def _init_csv(self, filepath: Path, fieldnames: List[str]):
        """Initialize CSV file with headers"""
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    def log_train(self, **kwargs):
        """
        Log training metrics
        
        Args:
            **kwargs: Metrics to log (must match train_fields)
            
        Example:
            >>> logger.log_train(
            ...     step=1000,
            ...     episode_return=50.0,
            ...     actor_loss=0.1,
            ...     critic_loss=0.05
            ... )
        """
        # Create row with None for missing fields
        row = {field: kwargs.get(field, None) for field in self.train_fields}
        
        # Append to CSV
        with open(self.train_log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.train_fields)
            writer.writerow(row)
    
    def log_eval(self, **kwargs):
        """
        Log evaluation metrics
        
        Args:
            **kwargs: Metrics to log (must match eval_fields)
            
        Example:
            >>> logger.log_eval(
            ...     step=20000,
            ...     mean_return=450.0,
            ...     std_return=20.0
            ... )
        """
        row = {field: kwargs.get(field, None) for field in self.eval_fields}
        
        with open(self.eval_log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.eval_fields)
            writer.writerow(row)
    
    def save_value_trajectory(self, trajectory: Dict, step: int):
        """
        Save value trajectory
        
        Args:
            trajectory: Trajectory dictionary from extract_value_trajectory
            step: Training step number
        """
        from src.eval.value_traj import save_value_trajectory
        save_value_trajectory(trajectory, self.run_dir, step)


if __name__ == "__main__":
    # Test logger
    print("Testing logger...")
    
    from pathlib import Path
    import shutil
    
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
    assert (Path('test_outputs/agent0/seed0/config.json')).exists()
    assert (Path('test_outputs/agent0/seed0/train_log.csv')).exists()
    assert (Path('test_outputs/agent0/seed0/eval_log.csv')).exists()
    
    print("✓ All files created")
    
    # Cleanup
    shutil.rmtree('test_outputs')
    
    print("\n✓ Logger tests passed!")
=======
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
>>>>>>> ef9b537a0fcad5f8b0cb77374192c2375ee31de1
