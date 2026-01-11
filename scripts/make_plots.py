# scripts/make_plots.py
from __future__ import annotations

from pathlib import Path

from src.plotting.aggregate import aggregate_metric
from src.plotting.plots import plot_curve, plot_losses


def main():
    project_root = Path(__file__).resolve().parents[1]
    runs_root = project_root / "outputs" / "runs"
    figs_root = project_root / "outputs" / "figures"
    figs_root.mkdir(parents=True, exist_ok=True)

    agents = ["agent0", "agent1", "agent2", "agent3", "agent4"]
    seeds = [0, 1, 2]

    for agent in agents:
        # Training return
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

    print(f"[OK] Figures generated in: {figs_root}")


if __name__ == "__main__":
    main()
