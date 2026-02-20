from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np

from .qsl import Trajectory, QSLMetrics


def plot_qsl_diagnostics(
    traj: Trajectory,
    metrics: QSLMetrics,
    path: str | Path,
    *,
    title_suffix: str | None = None,
) -> None:
    energies = traj.energies
    delta_e = traj.delta_e
    params = traj.params

    n_steps = len(energies) - 1
    step_idx = np.arange(1, n_steps + 1)
    iter_idx = np.arange(len(energies))

    finite_dt_star = np.where(np.isfinite(metrics.dt_star), metrics.dt_star, np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(iter_idx, energies, marker="o", linewidth=1.5)
    axes[0, 0].set_title("Energy per recorded iterate")
    axes[0, 0].set_xlabel("Recorded iterate index")
    axes[0, 0].set_ylabel(r"$E_k = \langle H \rangle_k$")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(iter_idx, delta_e, marker="s", color="tab:orange", linewidth=1.5)
    axes[0, 1].set_title("Energy uncertainty per iterate")
    axes[0, 1].set_xlabel("Recorded iterate index")
    axes[0, 1].set_ylabel(r"$\Delta E_k$")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(step_idx, metrics.raw_dt, marker="o", label=r"$\|\Delta\theta_k\|_2$ (raw step)")
    axes[1, 0].plot(step_idx, finite_dt_star, marker="^", label=r"$\delta t_k^*$ (QSL-consistent)")
    axes[1, 0].set_title("Step-size diagnostics")
    axes[1, 0].set_xlabel("Step k")
    axes[1, 0].set_ylabel("Step size")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(step_idx, metrics.s_n, marker="o", label=r"$S_N$")
    axes[1, 1].plot(step_idx, metrics.geo_prefix, marker="s", label=r"$\arccos(|\langle\psi_0|\psi_k\rangle|)$")
    axes[1, 1].plot(step_idx, metrics.s_raw, linestyle="--", color="gray", label="Cumulative with raw step")
    axes[1, 1].axhline(metrics.s0, color="tab:red", linestyle="--", linewidth=1.5, label=r"$S_0$")

    if metrics.n_min is not None:
        axes[1, 1].axvline(
            metrics.n_min,
            color="tab:green",
            linestyle=":",
            linewidth=1.5,
            label=f"first N with S_N >= S0: {metrics.n_min}",
        )

    axes[1, 1].set_title("QSL cumulative relation")
    axes[1, 1].set_xlabel("Step N")
    axes[1, 1].set_ylabel("Angle")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    if title_suffix:
        fig.suptitle(title_suffix, fontsize=12)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_param_sweep(
    sweep_results: List[Dict[str, Any]],
    path: str | Path,
    *,
    optimizer_name: str,
) -> None:
    if not sweep_results:
        return

    x = [entry["num_params"] for entry in sweep_results]
    actual_iters = [entry["actual_iterations"] for entry in sweep_results]
    qsl_iters = [
        entry["qsl_iterations"] if entry["qsl_iterations"] is not None else np.nan
        for entry in sweep_results
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, actual_iters, marker="o", label="Actual iterations")
    ax.plot(x, qsl_iters, marker="s", label="QSL iterations (N_min)")
    ax.set_xlabel("Number of parameters")
    ax.set_ylabel("Iterations")
    ax.set_title(f"Iteration counts vs parameters ({optimizer_name})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
