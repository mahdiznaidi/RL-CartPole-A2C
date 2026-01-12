# src/plotting/aggregate.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict
import re

import pandas as pd


@dataclass
class AggregatedCurve:
    step: List[int]
    mean: List[float]
    min_: List[float]
    max_: List[float]


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def aggregate_metric(
    runs_root: Path,
    agent_id: str,
    seeds: List[int],
    csv_name: str,
    step_col: str,
    metric_col: str,
) -> Optional[AggregatedCurve]:
    """
    Agrège un metric sur plusieurs seeds (mean + min/max shading).
    """
    frames = []
    for s in seeds:
        p = runs_root / agent_id / f"seed{s}" / csv_name
        df = _read_csv(p)
        if df is None or step_col not in df.columns or metric_col not in df.columns:
            continue
        sub = df[[step_col, metric_col]].dropna().copy()
        sub = sub.rename(columns={metric_col: f"{metric_col}_seed{s}"})
        frames.append(sub)

    if not frames:
        return None

    merged = frames[0]
    for f in frames[1:]:
        merged = pd.merge(merged, f, on=step_col, how="outer")

    merged = merged.sort_values(step_col).set_index(step_col)

    vals = merged.values
    mean = pd.DataFrame(vals).mean(axis=1, skipna=True).tolist()
    min_ = pd.DataFrame(vals).min(axis=1, skipna=True).tolist()
    max_ = pd.DataFrame(vals).max(axis=1, skipna=True).tolist()
    steps = merged.index.astype(int).tolist()

    return AggregatedCurve(step=steps, mean=mean, min_=min_, max_=max_)


def aggregate_episode_returns(
    runs_root: Path,
    agent_id: str,
    seeds: List[int],
    csv_name: str = "episode_log.csv",
    step_col: str = "step",
    metric_col: str = "episode_return",
    bin_size: int = 1000,
) -> Optional[AggregatedCurve]:
    """
    Aggregate episodic returns by binning steps (default: 1k steps).
    """
    frames = []
    for s in seeds:
        p = runs_root / agent_id / f"seed{s}" / csv_name
        df = _read_csv(p)
        if df is None or step_col not in df.columns or metric_col not in df.columns:
            continue
        sub = df[[step_col, metric_col]].dropna().copy()
        if sub.empty:
            continue
        sub["step_bin"] = ((sub[step_col] - 1) // bin_size + 1) * bin_size
        grouped = sub.groupby("step_bin", as_index=False)[metric_col].mean()
        grouped = grouped.rename(columns={"step_bin": step_col, metric_col: f"{metric_col}_seed{s}"})
        frames.append(grouped)

    if not frames:
        return None

    merged = frames[0]
    for f in frames[1:]:
        merged = pd.merge(merged, f, on=step_col, how="outer")

    merged = merged.sort_values(step_col).set_index(step_col)

    vals = merged.values
    mean = pd.DataFrame(vals).mean(axis=1, skipna=True).tolist()
    min_ = pd.DataFrame(vals).min(axis=1, skipna=True).tolist()
    max_ = pd.DataFrame(vals).max(axis=1, skipna=True).tolist()
    steps = merged.index.astype(int).tolist()

    return AggregatedCurve(step=steps, mean=mean, min_=min_, max_=max_)


def aggregate_value_means(
    runs_root: Path,
    agent_id: str,
    seeds: List[int],
) -> Optional[AggregatedCurve]:
    """
    Aggregate mean value along trajectory for each evaluation step.
    """
    frames = []
    for s in seeds:
        run_dir = runs_root / agent_id / f"seed{s}"
        if not run_dir.exists():
            continue
        rows: Dict[int, float] = {}
        for path in run_dir.glob("value_traj_step_*.csv"):
            match = re.search(r"value_traj_step_(\\d+)\\.csv", path.name)
            if not match:
                continue
            step = int(match.group(1))
            df = _read_csv(path)
            if df is None or "value" not in df.columns:
                continue
            rows[step] = float(df["value"].mean())
        if not rows:
            continue
        df_seed = pd.DataFrame(
            {"step": list(rows.keys()), f"value_mean_seed{s}": list(rows.values())}
        )
        frames.append(df_seed)

    if not frames:
        return None

    merged = frames[0]
    for f in frames[1:]:
        merged = pd.merge(merged, f, on="step", how="outer")

    merged = merged.sort_values("step").set_index("step")

    vals = merged.values
    mean = pd.DataFrame(vals).mean(axis=1, skipna=True).tolist()
    min_ = pd.DataFrame(vals).min(axis=1, skipna=True).tolist()
    max_ = pd.DataFrame(vals).max(axis=1, skipna=True).tolist()
    steps = merged.index.astype(int).tolist()

    return AggregatedCurve(step=steps, mean=mean, min_=min_, max_=max_)


def aggregate_value_trajectory(
    runs_root: Path,
    agent_id: str,
    seeds: List[int],
    eval_step: int,
) -> Optional[AggregatedCurve]:
    """
    Aggregate value along a trajectory for a specific evaluation step.
    """
    frames = []
    for s in seeds:
        path = runs_root / agent_id / f"seed{s}" / f"value_traj_step_{eval_step}.csv"
        df = _read_csv(path)
        if df is None or "step_in_episode" not in df.columns or "value" not in df.columns:
            continue
        sub = df[["step_in_episode", "value"]].dropna().copy()
        sub = sub.rename(columns={"value": f"value_seed{s}"})
        frames.append(sub)

    if not frames:
        return None

    merged = frames[0]
    for f in frames[1:]:
        merged = pd.merge(merged, f, on="step_in_episode", how="outer")

    merged = merged.sort_values("step_in_episode").set_index("step_in_episode")

    vals = merged.values
    mean = pd.DataFrame(vals).mean(axis=1, skipna=True).tolist()
    min_ = pd.DataFrame(vals).min(axis=1, skipna=True).tolist()
    max_ = pd.DataFrame(vals).max(axis=1, skipna=True).tolist()
    steps = merged.index.astype(int).tolist()

    return AggregatedCurve(step=steps, mean=mean, min_=min_, max_=max_)
