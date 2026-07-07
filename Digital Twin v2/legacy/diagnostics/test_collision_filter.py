"""
test_collision_filter.py
=========================

End-to-end demo/smoke test for the full redesigned scoring process:
BeamProgressScorer.combined_score() (target-closeness reward against the
REAL detector volume + electrode-collision penalty + chamber-exit penalty,
see beam_progress_score.py) built on the AABB-culled WallIndex
(electrode_geometry.py).

Builds the field map, generates a RANDOM voltage configuration (free
electrodes random within RANDOM_RANGE, fixed ones -- source/pipe/ground/
detector -- held at their real values), creates and flies the standard
500-particle beam, then runs the combined score and reports + plots what
happened: how many particles reached the target, left the chamber, or hit
an electrode on the way, plus how close the beam as a whole got to the
detector.

Run with:
    python test_collision_filter.py
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

from RK4_sim_basis import BasisFieldMap, IonSpecies, Beam, Trajectory, RK4Integrator
from beam_progress_score import make_beam, BeamProgressScorer
from electrode_geometry import build_wall_index

HERE = pathlib.Path(__file__).resolve().parent
N_ELECTRODES = 19

# Real detector volume, in workbench mm. Center confirmed directly: (75, 75,
# 405) -- same xz-plane (y=75) as the beam origination point [395, 75, 77].
# Box half-widths kept the same as the original optimizer.py DETECTOR_REGION
# estimate (x:6, y:6.5, z:2), just recentered.
DETECTOR_BBOX = (69.0, 81.0, 68.5, 81.5, 403.0, 407.0)

# Electrodes held at a fixed voltage -- matches optimizer.py's FIXED dict.
# Everything else (the lenses/benders/plates) gets a random voltage below.
FIXED = {
    1: 500.0,     # HV source
    19: -2000.0,  # Detector
    2: 0.0,       # pipe / outer housing
    4: 0.0, 5: 0.0, 7: 0.0, 8: 0.0,      # Einzel outer rings (grounded)
    13: 0.0, 14: 0.0, 16: 0.0, 17: 0.0,  # ground plates
}
RANDOM_RANGE = (-1000.0, 1000.0)

BEAM_N = 500
BEAM_START = [395.0, 75.0, 77.0]
DT = 5e-8
NUM_STEPS = 5000

# Electrodes checked by the collision filter: everything except the source
# (particles start there -- not a collision), the pipe/housing (its STL
# bounding box spans the whole chamber and is redundant with the outer-bbox
# "lost" check BeamProgressScorer already does), and the Detector (landing
# there is success, not a wall hit).
WALL_EXCLUDE = (1, 2, 19)
WALL_HIT_MARGIN = 1.5  # mm


def random_voltages(seed=None):
    rng = np.random.default_rng(seed)
    voltages = np.zeros(N_ELECTRODES)
    for i in range(1, N_ELECTRODES + 1):
        voltages[i - 1] = FIXED[i] if i in FIXED else rng.uniform(*RANDOM_RANGE)
    return voltages


def main():
    print("Loading field map from basis_electrode_*.csv ...")
    bfm = BasisFieldMap.from_directory(HERE, n_electrodes=N_ELECTRODES)

    voltages = random_voltages()
    print("\nRandom voltage configuration:")
    for i, v in enumerate(voltages, start=1):
        tag = " (fixed)" if i in FIXED else ""
        print(f"  electrode {i:2d}: {v:9.1f} V{tag}")
    bfm.set_voltages(voltages)

    species = IonSpecies(mass=28 * 1.66053906660e-27, charge=1.602176634e-19)  # Si+

    print(f"\nBuilding beam: N={BEAM_N}, start={BEAM_START}")
    start_positions, start_velocities = make_beam(
        N=BEAM_N, species=species, start_point=BEAM_START,
        mean_energy_eV=15.0, std_energy_eV=0.42466, half_angle_deg=15.0,
        seed=42,
    )

    beam = Beam(species=species, position=start_positions, velocity=start_velocities)
    trajectory = Trajectory(beam)
    integrator = RK4Integrator(bfm)

    print(f"Integrating ({NUM_STEPS} steps, dt={DT:.1e}s) -- this is the slow part...")
    integrator.integrate(trajectory, dt=DT, num_steps=NUM_STEPS)

    print(f"\nBuilding wall collision index (electrodes excluded: {WALL_EXCLUDE}) ...")
    wall_index = build_wall_index(HERE, exclude=WALL_EXCLUDE, target_spacing=2.0)
    print(f"  {len(wall_index.points)} surface sample points")

    scorer = BeamProgressScorer(
        bfm=bfm, Trajectory=trajectory, dt=DT, num_steps=NUM_STEPS,
        detector_bbox=DETECTOR_BBOX,
        wall_index=wall_index, wall_hit_margin=WALL_HIT_MARGIN,
    )
    print("\nScoring (combined_score: target-closeness reward + wall-hit + chamber-exit penalties) ...")
    result = scorer.combined_score(voltages)

    reached = result["reached_target"]
    lost = result["lost"]
    hit_wall = result["hit_wall"]
    neither = ~(reached | lost | hit_wall)

    print("\n=== RESULTS ===")
    print(f"combined_score       : {result['combined_score']:+.3f}   <- the new weighted score")
    print(f"mean target_reward   : {result['target_reward'].mean():.3f}  "
          f"(closest approach to detector, avg {result['target_distance'].mean():.1f}mm, "
          f"best {result['target_distance'].min():.1f}mm)")
    print(f"mean_progress        : {result['mean_progress']:.3f}  (old single-axis metric, for comparison)")
    print(f"survival_rate        : {result['survival_rate']:.3f}  ({reached.sum()}/{BEAM_N} actually entered the detector box)")
    print(f"lost (chamber exit)  : {lost.sum()}/{BEAM_N}")
    print(f"wall_hit_fraction    : {result['wall_hit_fraction']:.3f}  ({hit_wall.sum()}/{BEAM_N} hit an electrode)")
    print(f"still in flight      : {neither.sum()}/{BEAM_N}  (never reached/lost/hit within {NUM_STEPS} steps)")

    plot_result(trajectory, start_positions, wall_index, result, bfm)


def plot_result(trajectory, start_positions, wall_index, result, bfm):
    positions = np.array([s.position for s in trajectory.states])  # (T, N, 3)
    T, N = positions.shape[0], positions.shape[1]

    # RK4Integrator always runs the full num_steps, even after a particle
    # leaves the sampled field domain (where field() returns zero, so it
    # just drifts in a straight line forever). That drift dwarfs the actual
    # chamber on a shared axis. For plotting only, find each particle's
    # first step outside the field bbox and stop drawing there.
    lo = np.array([bfm.x.min(), bfm.y.min(), bfm.z.min()])
    hi = np.array([bfm.x.max(), bfm.y.max(), bfm.z.max()])
    outside = np.any((positions < lo) | (positions > hi), axis=2)  # (T, N)
    has_exit = outside.any(axis=0)
    plot_stop = np.where(has_exit, np.argmax(outside, axis=0), T - 1)  # (N,)

    # hit_wall drawn last / on top so the new feature is easy to spot even
    # for particles that also reached the target or left the chamber --
    # these categories are NOT mutually exclusive in the raw result (see
    # beam_progress_score.py), this ordering is just a plotting choice.
    outcome = np.full(N, "in flight", dtype=object)
    outcome[result["lost"]] = "lost (chamber exit)"
    outcome[result["reached_target"]] = "reached target"
    outcome[result["hit_wall"]] = "hit wall"

    colors = {
        "in flight": "royalblue",
        "lost (chamber exit)": "orange",
        "reached target": "green",
        "hit wall": "red",
    }

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Electrode walls: the same point cloud the collision filter checks
    # distance against, so what you see is literally what was tested.
    wp = wall_index.points
    stride = max(1, len(wp) // 8000)
    ax.scatter(wp[::stride, 0], wp[::stride, 1], wp[::stride, 2],
               color="gray", s=1, alpha=0.15, label="electrode walls (sampled)")

    for label, color in colors.items():
        idx = np.where(outcome == label)[0]
        if len(idx) == 0:
            continue
        for p in idx:
            end = plot_stop[p] + 1
            ax.plot(positions[:end, p, 0], positions[:end, p, 1], positions[:end, p, 2],
                     color=color, lw=0.5, alpha=0.5)
        ax.plot([], [], color=color, label=f"{label} ({len(idx)})")  # legend proxy

    ax.scatter(*start_positions[0], color="black", s=50, marker="x", label="beam start", depthshade=False)

    # Detector box, drawn as a small wireframe so it's clear what
    # target_reward/target_distance are actually measured against.
    dx0, dx1, dy0, dy1, dz0, dz1 = DETECTOR_BBOX
    corners = np.array([[x, y, z] for x in (dx0, dx1) for y in (dy0, dy1) for z in (dz0, dz1)])
    edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    for i, j in edges:
        ax.plot(*zip(corners[i], corners[j]), color="black", lw=1.0)
    ax.plot([], [], color="black", lw=1.0, label="detector box (target)")

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_title("Beam trajectories vs electrode walls (random voltage config)")
    ax.legend(loc="upper left", fontsize=8)

    out_path = HERE / "collision_filter_test.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
