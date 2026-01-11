# scripts/make_plots.py
from __future__ import annotations

from pathlib import Path

from src.plotting.aggregate import aggregate_metric_across_seeds, list_available_value_traj_files
from src.plotting.plots import plot_curve_with_band, plot_losses_two_curves, plot_value_trajectory_csv


def main():
    project_root = Path(__file__).resolve().parents[1]
    runs_root = project_root / "outputs" / "runs"
    figs_root = project_root / "outputs" / "figures"
    figs_root.mkdir(parents=True, exist_ok=True)

    agents = ["agent0", "agent1", "agent2", "agent3", "agent4"]
    seeds = [0, 1, 2]

    for agent in agents:
        # --- Training returns (si présent)
        train_return = aggregate_metric_across_seeds(
            runs_root=runs_root,
            agent_id=agent,
            seeds=seeds,
            csv_name="train_log.csv",
            step_col="step",
            metric_col="episode_return",
        )
        if train_return is not None:
            plot_curve_with_band(
                curve=train_return,
                title=f"{agent} - Training episodic return (undiscounted)",
                xlabel="env steps",
                ylabel="return",
                out_path=figs_root / f"{agent}_train_return.png",
            )

        # --- Evaluation returns (obligatoire normalement)
        eval_return = aggregate_metric_across_seeds(
            runs_root=runs_root,
            agent_id=agent,
            seeds=seeds,
            csv_name="eval_log.csv",
            step_col="step",
            metric_col="eval_return_mean",
        )
        if eval_return is not None:
            plot_curve_with_band(
                curve=eval_return,
                title=f"{agent} - Eval return (10 greedy episodes)",
                xlabel="env steps",
                ylabel="eval return",
                out_path=figs_root / f"{agent}_eval_return.png",
            )

        # --- Losses
        actor_loss = aggregate_metric_across_seeds(
            runs_root=runs_root,
            agent_id=agent,
            seeds=seeds,
            csv_name="train_log.csv",
            step_col="step",
            metric_col="actor_loss",
        )
        critic_loss = aggregate_metric_across_seeds(
            runs_root=runs_root,
            agent_id=agent,
            seeds=seeds,
            csv_name="train_log.csv",
            step_col="step",
            metric_col="critic_loss",
        )
        if actor_loss is not None or critic_loss is not None:
            plot_losses_two_curves(
                critic=critic_loss,
                actor=actor_loss,
                title=f"{agent} - Losses",
                out_path=figs_root / f"{agent}_losses.png",
            )

        # --- Value trajectory: on prend le 1er seed dispo, dernier fichier dispo
        traj_files = list_available_value_traj_files(runs_root, agent, seed=0)
        if traj_files:
            value_traj_csv = traj_files[-1]
            plot_value_trajectory_csv(
                value_traj_csv=value_traj_csv,
                title=f"{agent} - Value trajectory ({value_traj_csv.name})",
                out_path=figs_root / f"{agent}_value_traj.png",
            )

    print(f"[OK] Figures saved in: {figs_root}")


if __name__ == "__main__":
    main()
