"""
voltage_batch_filter.py
========================

Fast, RK4-based independent-sampling filter over a BATCH of Optuna-
suggested voltage arrays.

This is NOT meant to find the optimum -- that's what a real, SIMION-backed
optimizer run is for. It's meant to sieve a batch of candidates BEFORE any
of them cost a SIMION run: fly each one through the cheap RK4 model, score
it with BeamProgressScorer.combined_score() (target-closeness reward,
electrode-collision penalty, chamber-exit penalty), and use that to (a)
discard obviously-bad regions of voltage space for free, and (b) hand the
survivors' scores back as extra, SIMION-free training signal for the GP --
particularly valuable in regions where the GP itself is still uncertain
("zero gradient" spots it can't otherwise see without paying for SIMION).

COST NOTE: the RK4 integration itself (not the wall/collision check) is
the dominant per-candidate cost -- ~13s at full production fidelity
(500 particles x 5000 steps; see beam_progress_score.py / test_collision_
filter.py). VoltageBatchFilter defaults to a cheaper SCREENING fidelity
(150 particles x 2000 steps, ~1-2s/candidate) since the point here is
ruling regions in/out, not precision -- pass beam_n=500, num_steps=5000
for the full-fidelity analyzer settings if you want to trade speed for
accuracy on a smaller batch.

USAGE
-----
    from voltage_batch_filter import VoltageBatchFilter, random_voltage_batch

    filt = VoltageBatchFilter()                  # loads field map, beam, wall index once
    batch = random_voltage_batch(50, seed=0)      # or candidates from Optuna/the GP
    kept = filt.filter_batch(batch, top_k=10)      # flies + scores all 50, keeps the best 10

Run this file directly for a small demo.
"""

import pathlib
import time

import numpy as np

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
FIXED = {
    1: 500.0,     # HV source
    19: -2000.0,  # Detector
    2: 0.0,       # pipe / outer housing
    4: 0.0, 5: 0.0, 7: 0.0, 8: 0.0,      # Einzel outer rings (grounded)
    13: 0.0, 14: 0.0, 16: 0.0, 17: 0.0,  # ground plates
}
RANDOM_RANGE = (-1000.0, 1000.0)

# 1: source (particles start there). 2: pipe/housing (bbox spans the whole
# chamber -- redundant with the outer-bbox chamber-exit check anyway).
# 19: Detector (landing there is success, not a wall hit).
WALL_EXCLUDE = (1, 2, 19)


class VoltageBatchFilter:
    """
    Reusable, cheap-as-possible screener. Build once (loads the field map,
    fixed beam, and wall index -- the parts that DON'T change per
    candidate), then call score_batch()/filter_batch() with as many
    candidate voltage arrays as you like.
    """

    def __init__(self, directory=HERE, n_electrodes=N_ELECTRODES,
                 beam_n=150, beam_start=(395.0, 75.0, 77.0),
                 dt=5e-8, num_steps=2000,
                 detector_bbox=DETECTOR_BBOX, wall_exclude=WALL_EXCLUDE,
                 wall_target_spacing=2.0, wall_hit_margin=1.5,
                 target_weight=1.0, wall_weight=1.0, lost_weight=0.3,
                 target_scale=30.0, verbose=True):
        self.dt = dt
        self.num_steps = num_steps
        self.detector_bbox = detector_bbox
        self.weights = dict(target_weight=target_weight, wall_weight=wall_weight,
                             lost_weight=lost_weight, target_scale=target_scale)

        if verbose:
            print(f"[VoltageBatchFilter] loading field map ({n_electrodes} electrodes)...")
        self.bfm = BasisFieldMap.from_directory(directory, n_electrodes=n_electrodes)
        self.species = IonSpecies(mass=28 * 1.66053906660e-27, charge=1.602176634e-19)

        if verbose:
            print(f"[VoltageBatchFilter] building fixed beam (N={beam_n})...")
        self.start_positions, self.start_velocities = make_beam(
            N=beam_n, species=self.species, start_point=list(beam_start),
            mean_energy_eV=15.0, std_energy_eV=0.42466, half_angle_deg=15.0,
            seed=42,
        )

        if verbose:
            print("[VoltageBatchFilter] building wall collision index...")
        self.wall_index = build_wall_index(directory, exclude=wall_exclude,
                                            target_spacing=wall_target_spacing)
        self.wall_hit_margin = wall_hit_margin
        self.verbose = verbose

    def _fly(self, voltages):
        self.bfm.set_voltages(voltages)
        beam = Beam(species=self.species, position=self.start_positions,
                    velocity=self.start_velocities)
        trajectory = Trajectory(beam)
        RK4Integrator(self.bfm).integrate(trajectory, dt=self.dt, num_steps=self.num_steps)
        return trajectory

    def score_one(self, voltages):
        """Fly + score a single voltage array. Returns a small summary dict
        (not the full per-particle arrays -- call BeamProgressScorer directly
        if you need those for one specific candidate)."""
        voltages = np.asarray(voltages, dtype=float)
        trajectory = self._fly(voltages)
        scorer = BeamProgressScorer(
            bfm=self.bfm, Trajectory=trajectory, dt=self.dt, num_steps=self.num_steps,
            detector_bbox=self.detector_bbox,
            wall_index=self.wall_index, wall_hit_margin=self.wall_hit_margin,
        )
        result = scorer.combined_score(voltages, **self.weights)
        return {
            "voltages": voltages,
            "combined_score": result["combined_score"],
            "mean_progress": result["mean_progress"],
            "survival_rate": result["survival_rate"],
            "wall_hit_fraction": result["wall_hit_fraction"],
            "lost_fraction": float(result["lost"].mean()),
        }

    def score_batch(self, voltage_arrays):
        """
        voltage_arrays : (M, n_electrodes) array_like
        Returns a list of M summary dicts (see score_one), same order as input.
        """
        voltage_arrays = np.asarray(voltage_arrays, dtype=float)
        results = []
        for i, v in enumerate(voltage_arrays):
            t0 = time.time()
            summary = self.score_one(v)
            summary["elapsed_s"] = time.time() - t0
            results.append(summary)
            if self.verbose:
                print(f"  [{i+1}/{len(voltage_arrays)}] combined_score={summary['combined_score']:+.3f}  "
                      f"progress={summary['mean_progress']:.3f}  survival={summary['survival_rate']:.3f}  "
                      f"wall_hit={summary['wall_hit_fraction']:.3f}  ({summary['elapsed_s']:.1f}s)")
        return results

    def filter_batch(self, voltage_arrays, min_combined_score=None, top_k=None):
        """
        score_batch() then keep only candidates with combined_score >=
        min_combined_score (if given) and/or the top_k highest-scoring (if
        given). Returns the kept summaries, sorted best-first -- these are
        the ones worth spending real SIMION time on / feeding to the GP.
        """
        results = self.score_batch(voltage_arrays)
        results.sort(key=lambda r: r["combined_score"], reverse=True)
        if min_combined_score is not None:
            results = [r for r in results if r["combined_score"] >= min_combined_score]
        if top_k is not None:
            results = results[:top_k]
        return results


def random_voltage_batch(m, seed=None):
    """M random candidate arrays: free electrodes random in RANDOM_RANGE, fixed ones held."""
    rng = np.random.default_rng(seed)
    batch = np.zeros((m, N_ELECTRODES))
    for i in range(1, N_ELECTRODES + 1):
        if i in FIXED:
            batch[:, i - 1] = FIXED[i]
        else:
            batch[:, i - 1] = rng.uniform(*RANDOM_RANGE, size=m)
    return batch


if __name__ == "__main__":
    M = 6
    print(f"Demo: scoring a random batch of {M} voltage arrays "
          f"(screening fidelity: 150 particles x 2000 steps)\n")

    filt = VoltageBatchFilter()
    batch = random_voltage_batch(M, seed=0)

    t0 = time.time()
    kept = filt.filter_batch(batch, min_combined_score=0.0)
    print(f"\nTotal batch time: {time.time()-t0:.1f}s for {M} candidates")

    print(f"\n{len(kept)}/{M} candidates kept (combined_score >= 0.0), best first:")
    for r in kept:
        print(f"  combined_score={r['combined_score']:+.3f}  survival_rate={r['survival_rate']:.3f}  "
              f"wall_hit_fraction={r['wall_hit_fraction']:.3f}")
