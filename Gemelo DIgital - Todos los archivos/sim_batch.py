"""
sim_batch.py
=============

Batched replacement for digital-twins-beamline-mit/sim.py's core loop.
Does NOT modify sim.py, RK4_sim_basis.py, or optimizer.py -- those stay as
backups. See RK4_sim_basis_batch.py for what changed and why.

sim.py samples M candidate voltage arrays from Optuna's sampler, then for
each one separately: field_map.set_voltages(...) (overwrites the ONE
field the map can hold) + a full RK4Integrator.integrate() over the
500-particle beam -- M sequential integrations.

This script samples the same M candidates, but builds all M combined
fields at once (BatchBasisFieldMap.set_voltages_batch) and flies every
config's particles in ONE RK4Integrator.integrate() call.

SCORING: one BeamProgressScorer call for the whole chunk, not a loop
-----------------------------------------------------------------------
BeamProgressScorer's config-independent setup (bfm, wall_index,
detector_bbox, bbox/target) is identical for every config in a chunk --
and since every config replays the SAME base beam (make_batch_beam just
tiles it), start_coord is identical too. Nothing inside score()/
combined_score() actually depends on which config a particle belongs to;
they're already fully vectorized across whatever "particle axis" they're
given. So instead of looping "split into per-config trajectories, build a
scorer, score" M times, score() is called ONCE on the whole chunk's
(configs*N) particles, and the returned PER-PARTICLE arrays
(target_reward, hit_wall, lost) are reshaped to (n_configs, N) and reduced
with .mean(axis=1) to get back per-config numbers -- valid because
make_batch_beam lays particles out as contiguous per-config blocks
(config_indices = repeat(arange(n_configs), N)).

This also fixes the earlier memory bug, not just papers over it with
chunking: split_batch_trajectory used to materialize a FULL duplicate copy
of the chunk's trajectory data (one copy per config, all held at once,
on top of the original) purely so each config could get its own Trajectory
object for a separate scorer call -- memory spent on nothing but
re-arranging data that already existed. Scoring the whole chunk in one
call needs exactly one array-ification of the trajectory (score()'s own
positions = np.array(...) step, unavoidable to vectorize over at all) --
not one plus M redundant copies of the same data.
split_batch_trajectory is kept in RK4_sim_basis_batch.py as a utility (e.g.
for pulling out ONE config's Trajectory to plot), just not used in this
hot path anymore.

MEMORY: CHUNK_SIZE, not just M
--------------------------------
Peak memory per chunk is still roughly 2 x CHUNK_SIZE x (per-config
trajectory size) -- the trajectory itself (from integration) plus the one
array-ification score() needs -- e.g. ~1.2GB for CHUNK_SIZE=5 at this
project's default fidelity (5001 steps x 500 particles x ~120MB/config).
Still chunk by CHUNK_SIZE rather than processing all M at once, to keep
that bounded independent of total M; tune CHUNK_SIZE to your machine's RAM
using that formula.

Run with:
    python sim_batch.py
"""

import pathlib
import time

import numpy as np
import optuna

import optimizer as op
from RK4_sim_basis import IonSpecies
from RK4_sim_basis_batch import (
    BatchBasisFieldMap, make_batch_beam, BatchTrajectory, BatchRK4Integrator,
)
from beam_progress_score import make_beam, BeamProgressScorer
from electrode_geometry import build_wall_index
from cost_planner import plan_batch
from starting_point import inject_starting_point

HERE = pathlib.Path(__file__).resolve().parent
N_ELECTRODES = 19
# 1e-8, not the historical 5e-8 -- see orchestrator.py's DT comment (beam
# is ~470 eV after the corrected acceleration units, 5e-8 was ~2.9mm/step).
DT = 1e-8

# How many of the RK4-screened candidates you actually intend to promote
# to real SIMION runs this iteration -- a resource decision (SIMION time),
# not derivable from RK4 cost alone. M (how many to screen) and the
# screening fidelity (BEAM_N, NUM_STEPS) are DERIVED from this via
# cost_planner.plan_batch(), using real measured RK4/SIMION costs -- see
# that module for the numbers and reasoning ("speed not accuracy" for the
# RK4 stage, with a physics-justified floor on NUM_STEPS).
SIMION_BUDGET_K = 20
_PLAN = plan_batch(simion_budget_k=SIMION_BUDGET_K)
M = _PLAN["M"]
BEAM_N = _PLAN["rk4_particles"]
NUM_STEPS = _PLAN["rk4_steps"]

# Per-config trajectory memory is now ~(NUM_STEPS * BEAM_N * 3 * 8 * 2) bytes
# -- at the cost-planner's default screening fidelity (50 particles, 2500
# steps) that's ~6MB/config, vs ~120MB/config at the old full-fidelity
# defaults (500, 5000). CHUNK_SIZE can scale up accordingly; see
# sim_batch.py's docstring for the "2 x CHUNK_SIZE x per-config size" peak
# memory formula.
CHUNK_SIZE = 50

DETECTOR_BBOX = (69.0, 81.0, 68.5, 81.5, 403.0, 407.0)
WALL_EXCLUDE = (1, 2, 19)  # source, pipe/housing, Detector -- see electrode_geometry.py

# Passed to combined_score() AND reused below to reduce per-particle arrays
# to per-config scores -- kept as one dict so the two stay in sync.
#
# target_scale was 30.0 -- wrong length scale (matched lens feature size,
# not actual observed miss distance). Every batch this session showed
# identical combined_score to 3 decimals regardless of voltage draw
# because exp(-235..350mm / 30mm) rounds to ~0 for every real candidate,
# contributing zero signal. 150mm keeps the reward discriminating across
# the 200-350mm range actually being sampled -- see beam_progress_score.py
# combined_score()'s target_scale docstring for the full diagnosis.
SCORE_WEIGHTS = dict(target_weight=1.0, wall_weight=1.0, lost_weight=0.3, target_scale=150.0)


def sample_voltage_batch(m, seed=None):
    """Mirrors sim.py's suggested_values construction: FIXED electrodes held
    at their real value, OPTIMIZE ones drawn from Optuna's own sampler
    against the existing study (independent sampling, not a full trial)."""
    study = optuna.create_study(direction=op.DIRECTION,
                                 storage=f"sqlite:///{op.RESULTS_DB}",
                                 study_name=op.STUDY_NAME, load_if_exists=True)
    sampler = study.sampler
    dist = optuna.distributions.FloatDistribution(low=-1000.0, high=1000.0)

    all_volts = {**op.FIXED, **op.OPTIMIZE}
    batch = np.zeros((m, N_ELECTRODES))
    for i in range(m):
        for key, value in all_volts.items():
            if isinstance(value, float):
                batch[i, key - 1] = value
            else:
                batch[i, key - 1] = sampler.sample_independent(
                    study, trial=None, param_name="x", param_distribution=dist)
    return batch


def score_chunk(bfm, wall_index, voltages_chunk, start_positions, start_velocities, species):
    """
    Run the full batched pipeline (build fields, integrate, score) for one
    chunk of candidates in a SINGLE BeamProgressScorer call -- see the
    module docstring for why this is both faster (no M-times-repeated
    scorer construction / wall-index queries) and the actual memory fix
    (no split_batch_trajectory duplication), not just chunking around it.

    Returns
    -------
    (n_configs,) ndarray of combined_score per config.
    """
    n_configs = voltages_chunk.shape[0]
    n = start_positions.shape[0]  # particles per config

    bfm.set_voltages_batch(voltages_chunk)

    beam, config_indices = make_batch_beam(species, start_positions, start_velocities, n_configs)
    trajectory = BatchTrajectory(beam, config_indices)
    BatchRK4Integrator(bfm, config_indices).integrate(trajectory, dt=DT, num_steps=NUM_STEPS)

    # ONE scorer call for every particle in the chunk at once -- valid
    # because bfm/wall_index/detector_bbox and the (replicated) beam are
    # identical across configs, so nothing here actually depends on config
    # identity except how the resulting per-particle arrays get grouped
    # back up afterward.
    #
    # wall_check_midpoints=False, wall_check_stride=3: profiling found the
    # wall-collision check was ~86% of combined_score()'s cost (see
    # cost_planner.py). Measured across 3 different random voltage batches,
    # this setting gives an IDENTICAL wall_hit_fraction/combined_score to
    # the thorough default (midpoints=True, stride=1) at ~4.2x less
    # scoring time -- stride=5 was tested too and DID start losing signal,
    # so 3 is the validated floor, not an arbitrary choice.
    scorer = BeamProgressScorer(
        bfm=bfm, Trajectory=trajectory, dt=DT, num_steps=NUM_STEPS,
        detector_bbox=DETECTOR_BBOX, wall_index=wall_index, wall_hit_margin=1.5,
        wall_check_midpoints=False, wall_check_stride=3,
    )
    result = scorer.combined_score(voltages_chunk, **SCORE_WEIGHTS)

    # Reduce (n_configs*n,) per-particle arrays back to (n_configs,) --
    # valid because make_batch_beam lays particles out as n_configs
    # contiguous blocks of n (config_indices = repeat(arange(n_configs), n)).
    target_reward = result["target_reward"].reshape(n_configs, n).mean(axis=1)
    wall_hit_fraction = result["hit_wall"].reshape(n_configs, n).mean(axis=1)
    lost_fraction = result["lost"].reshape(n_configs, n).mean(axis=1)

    return (
        SCORE_WEIGHTS["target_weight"] * target_reward
        - SCORE_WEIGHTS["wall_weight"] * wall_hit_fraction
        - SCORE_WEIGHTS["lost_weight"] * lost_fraction
    )


def main():
    print(f"Cost-planner: SIMION budget K={SIMION_BUDGET_K} -> screen M={M} candidates "
          f"({BEAM_N} particles x {NUM_STEPS} steps each, {_PLAN['subset_fraction']*100:.1f}% "
          f"would be promoted) -- predicted {_PLAN['predicted_screening_time_s']:.1f}s RK4 + "
          f"{_PLAN['predicted_simion_time_s']:.1f}s SIMION if K were run for real\n")

    print(f"Sampling {M} candidate voltage arrays from Optuna's sampler...")
    voltages_batch = sample_voltage_batch(M)

    # Option #3: re-center part of the batch on the derived starting point,
    # if bender_field_analysis.py has produced one -- see starting_point.py.
    n_sp, _ = inject_starting_point(voltages_batch, op.OPTIMIZE)
    if n_sp:
        print(f"Injected derived starting point + {n_sp - 1} perturbations around it "
              f"(see derived_starting_point.json)")

    print("Loading field map...")
    bfm = BatchBasisFieldMap.from_directory(HERE, n_electrodes=N_ELECTRODES)

    print("Building wall collision index (reused across all chunks)...")
    wall_index = build_wall_index(HERE, exclude=WALL_EXCLUDE, target_spacing=2.0)

    species = IonSpecies(mass=28 * 1.66053906660e-27, charge=1.602176634e-19)
    start_positions, start_velocities = make_beam(
        N=BEAM_N, species=species, start_point=[395.0, 75.0, 77.0],
        mean_energy_eV=15.0, std_energy_eV=0.42466, half_angle_deg=15.0, seed=42,
    )

    n_chunks = int(np.ceil(M / CHUNK_SIZE))
    print(f"\nProcessing {M} candidates in {n_chunks} chunk(s) of up to {CHUNK_SIZE} "
          f"({CHUNK_SIZE * BEAM_N} particles/chunk)...")

    scores = np.empty(M)
    t0 = time.time()
    for c in range(n_chunks):
        lo, hi = c * CHUNK_SIZE, min((c + 1) * CHUNK_SIZE, M)
        chunk_t0 = time.time()
        scores[lo:hi] = score_chunk(bfm, wall_index, voltages_batch[lo:hi],
                                     start_positions, start_velocities, species)
        print(f"  chunk {c+1}/{n_chunks} (configs {lo}-{hi-1}): {time.time()-chunk_t0:.1f}s")
    total = time.time() - t0

    order = np.argsort(scores)[::-1]
    print(f"\n=== Top {min(10, M)} of {M} candidates by combined_score ===")
    for rank, idx in enumerate(order[:10], 1):
        print(f"  #{rank}: config {idx:3d}  combined_score={scores[idx]:+.3f}")

    print(f"\nTotal: {total:.1f}s for {M} candidates ({total/M*1000:.1f}ms/candidate average)")
    print(f"(cost-planner predicted {_PLAN['predicted_screening_time_s']:.1f}s for this M/fidelity; "
          f"for comparison, {SIMION_BUDGET_K} real SIMION runs alone would cost "
          f"~{SIMION_BUDGET_K * 5.77:.1f}s)")


if __name__ == "__main__":
    main()
