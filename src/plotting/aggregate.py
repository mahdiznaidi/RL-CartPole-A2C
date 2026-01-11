# src/plotting/aggregate.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class AggregatedCurve:
    step: List[int]
    mean: List[float]
    min_: List[float]
    max_: List[float]


def _read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def aggregate_metric_across_seeds(
    runs_root: Path,
    agent_id: str,
    seeds: List[int],
    csv_name: str,
    step_col: str,
    metric_col: str,
) -> Optional[AggregatedCurve]:
    """
    Lit outputs/runs/<agent_id>/seedX/<csv_name>
    Aligne par step, puis calcule mean/min/max sur seeds.
    """
    frames = []
    for s in seeds:
        p = Path(runs_root) / agent_id / f"seed{s}" / csv_name
        df = _read_csv_if_exists(p)
        if df is None or step_col not in df.columns or metric_col not in df.columns:
            continue
        sub = df[[step_col, metric_col]].dropna().copy()
        sub = sub.rename(columns={metric_col: f"{metric_col}_seed{s}"})
        frames.append(sub)

    if not frames:
        return None

    # Merge sur step
    merged = frames[0]
    for f in frames[1:]:
        merged = pd.merge(merged, f, on=step_col, how="outer")

    merged = merged.sort_values(step_col)
    merged = merged.set_index(step_col)

    # Calcul stats en ignorant NaN
    values = merged.values
    mean = pd.DataFrame(values).mean(axis=1, skipna=True).tolist()
    min_ = pd.DataFrame(values).min(axis=1, skipna=True).tolist()
    max_ = pd.DataFrame(values).max(axis=1, skipna=True).tolist()
    steps = merged.index.astype(int).tolist()

    return AggregatedCurve(step=steps, mean=mean, min_=min_, max_=max_)


def list_available_value_traj_files(runs_root: Path, agent_id: str, seed: int) -> List[Path]:
    """
    Renvoie les fichiers value_traj_eval_*.csv si existants.
    """
    run_dir = Path(runs_root) / agent_id / f"seed{seed}"
    if not run_dir.exists():
        return []
    return sorted(run_dir.glob("value_traj_eval_*.csv"))
