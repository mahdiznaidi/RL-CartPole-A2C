# scripts/make_plots.py
from __future__ import annotations

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.plotting.aggregate import (
    aggregate_metric,
    aggregate_episode_returns,
    aggregate_value_means,
    aggregate_value_trajectory,
)
from src.plotting.plots import plot_curve, plot_losses, plot_value_trajectory


def main():
    project_root = Path(__file__).resolve().parents[1]
    runs_root = project_root / "outputs" / "runs"
    figs_root = project_root / "outputs" / "figures"
    figs_root.mkdir(parents=True, exist_ok=True)

    agents = ["agent0", "agent1", "agent2", "agent3", "agent4"]
    seeds = [0, 1, 2]

    for agent in agents:
        # Training return (binned episode returns)
        tr = aggregate_episode_returns(
            runs_root=runs_root,
            agent_id=agent,
            seeds=seeds,
        )
        if tr is None:
            tr = aggregate_metric(
                runs_root=runs_root,
                agent_id=agent,
                seeds=seeds,
                csv_name="train_log.csv",
                step_col="step",
                metric_col="episode_return",
            )
        if tr is not None:
            plot_curve(tr, f"{agent} - Training episodic return", "return", figs_root / f"{agent}_train_return.png")

        # Eval return (on lit eval_return_mean si présent, sinon mean_return)
        ev = aggregate_metric(
            runs_root=runs_root,
            agent_id=agent,
            seeds=seeds,
            csv_name="eval_log.csv",
            step_col="step",
            metric_col="eval_return_mean",
        )
        if ev is None:
            ev = aggregate_metric(
                runs_root=runs_root,
                agent_id=agent,
                seeds=seeds,
                csv_name="eval_log.csv",
                step_col="step",
                metric_col="mean_return",
            )
        if ev is not None:
            plot_curve(ev, f"{agent} - Eval return (greedy)", "eval return", figs_root / f"{agent}_eval_return.png")

        # Losses
        actor = aggregate_metric(
            runs_root=runs_root,
            agent_id=agent,
            seeds=seeds,
            csv_name="train_log.csv",
            step_col="step",
            metric_col="actor_loss",
        )
        critic = aggregate_metric(
            runs_root=runs_root,
            agent_id=agent,
            seeds=seeds,
            csv_name="train_log.csv",
            step_col="step",
            metric_col="critic_loss",
        )
        if actor is not None or critic is not None:
            plot_losses(actor, critic, f"{agent} - Losses", figs_root / f"{agent}_losses.png")

        # Value function (mean along trajectory across evals)
        val_mean = aggregate_value_means(
            runs_root=runs_root,
            agent_id=agent,
            seeds=seeds,
        )
        if val_mean is not None:
            plot_curve(
                val_mean,
                f"{agent} - Value mean across evals",
                "value (mean along traj)",
                figs_root / f"{agent}_value_mean.png",
            )

        # Value trajectory for latest evaluation
        if val_mean is not None and val_mean.step:
            latest_step = max(val_mean.step)
            traj_curve = aggregate_value_trajectory(
                runs_root=runs_root,
                agent_id=agent,
                seeds=seeds,
                eval_step=latest_step,
            )
            if traj_curve is not None:
                plot_value_trajectory(
                    traj_curve,
                    f"{agent} - Value trajectory (eval {latest_step})",
                    figs_root / f"{agent}_value_traj_{latest_step}.png",
                )

    print(f"[OK] Figures generated in: {figs_root}")


if __name__ == "__main__":
    main()
