# src/plotting/plots.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from .aggregate import AggregatedCurve


def _save(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_curve(curve: AggregatedCurve, title: str, ylabel: str, out_path: Path) -> None:
    plt.figure()
    plt.plot(curve.step, curve.mean, label="mean")
    plt.fill_between(curve.step, curve.min_, curve.max_, alpha=0.2, label="min/max")
    plt.title(title)
    plt.xlabel("env steps")
    plt.ylabel(ylabel)
    plt.legend()
    _save(out_path)


def plot_losses(actor: Optional[AggregatedCurve], critic: Optional[AggregatedCurve], title: str, out_path: Path) -> None:
    plt.figure()
    if actor is not None:
        plt.plot(actor.step, actor.mean, label="actor_loss")
        plt.fill_between(actor.step, actor.min_, actor.max_, alpha=0.2)
    if critic is not None:
        plt.plot(critic.step, critic.mean, label="critic_loss")
        plt.fill_between(critic.step, critic.min_, critic.max_, alpha=0.2)
    plt.title(title)
    plt.xlabel("env steps")
    plt.ylabel("loss")
    plt.legend()
    _save(out_path)


def plot_value_trajectory(curve: AggregatedCurve, title: str, out_path: Path) -> None:
    plt.figure()
    plt.plot(curve.step, curve.mean, label="mean")
    plt.fill_between(curve.step, curve.min_, curve.max_, alpha=0.2, label="min/max")
    plt.title(title)
    plt.xlabel("step in episode")
    plt.ylabel("value")
    plt.legend()
    _save(out_path)
