# src/plotting/plots.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from .aggregate import AggregatedCurve


def _save_fig(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_curve_with_band(
    curve: AggregatedCurve,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    plt.figure()
    plt.plot(curve.step, curve.mean, label="mean")
    plt.fill_between(curve.step, curve.min_, curve.max_, alpha=0.2, label="min/max")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    _save_fig(out_path)


def plot_losses_two_curves(
    critic: Optional[AggregatedCurve],
    actor: Optional[AggregatedCurve],
    title: str,
    out_path: Path,
) -> None:
    plt.figure()
    if actor is not None:
        plt.plot(actor.step, actor.mean, label="actor_loss_mean")
        plt.fill_between(actor.step, actor.min_, actor.max_, alpha=0.2)
    if critic is not None:
        plt.plot(critic.step, critic.mean, label="critic_loss_mean")
        plt.fill_between(critic.step, critic.min_, critic.max_, alpha=0.2)
    plt.title(title)
    plt.xlabel("env steps")
    plt.ylabel("loss")
    plt.legend()
    _save_fig(out_path)


def plot_value_trajectory_csv(value_traj_csv: Path, title: str, out_path: Path) -> None:
    """
    Plot V(s) sur une trajectoire (un fichier csv unique).
    CSV attendu: colonnes au moins ["t","v"] (ou "step","v").
    """
    import pandas as pd

    df = pd.read_csv(value_traj_csv)
    t_col = "t" if "t" in df.columns else ("step" if "step" in df.columns else None)
    if t_col is None or "v" not in df.columns:
        return

    plt.figure()
    plt.plot(df[t_col].values, df["v"].values)
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("V(s)")
    _save_fig(out_path)
