"""
plot_cycle_best.py
=====================

Pulls the "best" (RK4 rank=1) promoted candidate from each cycle
(iteration) of orchestrator.py's persisted study, re-flies all of them
together in one batched RK4 call (full fidelity -- these are just the 6
"best of cycle" candidates, not a big screening batch), and plots the
resulting beam paths together in the experiment space, color-coded by
cycle, so improvement (or lack of it) across cycles is visible directly.

Note: all real SIMION values on record are tied at 0 hits, so "best of
cycle" here means "the candidate RK4 ranked #1 that cycle" (the one that
was actually promoted first) -- not "best real outcome", since there's no
variation in real outcome to rank by yet.

Trials from BEFORE the rk4_score/iteration user_attrs existed (the very
first small-scale test, trial #s 0-5) aren't taggable by cycle and are
excluded from this plot -- see report_orchestrator_run.py for a full
trial-by-trial dump including those.

Run with:
    python plot_cycle_best.py
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import optuna

from RK4_sim_basis import IonSpecies
from RK4_sim_basis_batch import BatchBasisFieldMap, make_batch_beam, BatchTrajectory, BatchRK4Integrator
from beam_progress_score import make_beam
from electrode_geometry import build_wall_index
from orchestrator import STUDY_NAME, RESULTS_DB, N_ELECTRODES

optuna.logging.set_verbosity(optuna.logging.WARNING)

HERE = pathlib.Path(__file__).resolve().parent
DETECTOR_BBOX = (69.0, 81.0, 68.5, 81.5, 403.0, 407.0)
WALL_EXCLUDE = (1, 2, 19)

DT = 5e-8
NUM_STEPS = 5000  # full fidelity -- only 6 candidates, cheap enough


def collect_cycle_bests():
    """Group COMPLETE trials into (run, iteration) cycles using trial-number
    order (iteration numbers reset to 1 at the start of each orchestrate()
    call, so a same-or-lower iteration number than the previous trial marks
    a new run), then take the rk4_rank_in_iteration==1 trial from each."""
    study = optuna.load_study(study_name=STUDY_NAME, storage=f"sqlite:///{RESULTS_DB}")
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
              and "iteration" in t.user_attrs]
    trials.sort(key=lambda t: t.number)

    cycles = []  # list of (run_id, iteration, trial)
    run_id = 0
    prev_iter = None
    for t in trials:
        it = t.user_attrs["iteration"]
        # A DECREASE (not just a repeat -- same iteration has several
        # trials, ranks 1..K) marks the start of a new orchestrate() call.
        if prev_iter is not None and it < prev_iter:
            run_id += 1
        prev_iter = it
        if t.user_attrs.get("rk4_rank_in_iteration") == 1:
            cycles.append((run_id, it, t))
    return cycles


def main():
    cycles = collect_cycle_bests()
    print(f"Found {len(cycles)} cycles with a tagged rank-1 candidate:")
    for run_id, it, t in cycles:
        print(f"  run {run_id}, iteration {it}: trial #{t.number}, rk4_score={t.user_attrs['rk4_score']:+.6f}")

    voltages_batch = np.zeros((len(cycles), N_ELECTRODES))
    labels = []
    import optimizer as op
    for row, (run_id, it, t) in enumerate(cycles):
        for k, v in op.FIXED.items():
            voltages_batch[row, k - 1] = v
        for k in op.OPTIMIZE:
            voltages_batch[row, k - 1] = t.params[f"V{k}"]
        labels.append(f"run{run_id} it{it} (#{t.number})")

    print("\nLoading field map...")
    bfm = BatchBasisFieldMap.from_directory(HERE, n_electrodes=N_ELECTRODES)
    print("Building wall collision index...")
    wall_index = build_wall_index(HERE, exclude=WALL_EXCLUDE, target_spacing=2.0)
    species = IonSpecies(mass=28 * 1.66053906660e-27, charge=1.602176634e-19)

    n = voltages_batch.shape[0]
    start_positions, start_velocities = make_beam(
        N=500, species=species, start_point=[395.0, 75.0, 77.0],
        mean_energy_eV=15.0, std_energy_eV=0.42466, half_angle_deg=15.0, seed=42,
    )

    print(f"Flying {n} 'best of cycle' candidates together (full fidelity, {NUM_STEPS} steps)...")
    bfm.set_voltages_batch(voltages_batch)
    beam, config_indices = make_batch_beam(species, start_positions, start_velocities, n)
    trajectory = BatchTrajectory(beam, config_indices)
    BatchRK4Integrator(bfm, config_indices).integrate(trajectory, dt=DT, num_steps=NUM_STEPS)

    positions = np.array([s.position for s in trajectory.states])  # (T, n*500, 3)

    lo = np.array([bfm.x.min(), bfm.y.min(), bfm.z.min()])
    hi = np.array([bfm.x.max(), bfm.y.max(), bfm.z.max()])

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    wp = wall_index.points
    stride = max(1, len(wp) // 6000)
    ax.scatter(wp[::stride, 0], wp[::stride, 1], wp[::stride, 2],
               color="gray", s=1, alpha=0.12, label="electrode walls (sampled)")

    dx0, dx1, dy0, dy1, dz0, dz1 = DETECTOR_BBOX
    corners = np.array([[x, y, z] for x in (dx0, dx1) for y in (dy0, dy1) for z in (dz0, dz1)])
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    for i, j in edges:
        ax.plot(*zip(corners[i], corners[j]), color="black", lw=1.2)
    ax.plot([], [], color="black", lw=1.2, label="detector box (target)")

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, n))
    N_PER = 500
    for row in range(n):
        mask = config_indices == row
        pos_row = positions[:, mask, :]  # (T, 500, 3)

        outside = np.any((pos_row < lo) | (pos_row > hi), axis=2)
        has_exit = outside.any(axis=0)
        plot_stop = np.where(has_exit, np.argmax(outside, axis=0), pos_row.shape[0] - 1)

        # plot a subsample of particles per cycle to keep the figure legible
        particle_idx = np.linspace(0, N_PER - 1, 40, dtype=int)
        for p in particle_idx:
            end = plot_stop[p] + 1
            ax.plot(pos_row[:end, p, 0], pos_row[:end, p, 1], pos_row[:end, p, 2],
                    color=colors[row], lw=0.4, alpha=0.35)
        ax.plot([], [], color=colors[row], label=labels[row])

    ax.scatter(*start_positions[0], color="black", s=50, marker="x", label="beam start", depthshade=False)

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_title("Best-of-cycle candidates across the orchestrator run (color = cycle)")
    ax.legend(loc="upper left", fontsize=7)

    out_path = HERE / "cycle_best_comparison.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
