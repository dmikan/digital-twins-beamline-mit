"""
starting_point.py
==================

Shared access to the physically-informed starting point derived by
bender_field_analysis.py (SESSION_HANDOFF.txt option #3), plus the one
piece of logic both consumers need: re-centering a cheap RK4-screening
batch around it instead of sampling uniform [-1000, 1000] everywhere.

Why injection into the batch, not study.enqueue_trial(): enqueue_trial
only affects study.ask(), and the orchestrator's default TPE path never
calls ask() -- it generates screening candidates via cheap
sample_independent() draws. Injecting the starting point (and Gaussian
perturbations around it) directly into the screening batch guarantees it
actually gets flown, and -- if it screens well -- promoted to real SIMION
and told to the study, at which point the sampler's own history takes
over. Works identically for the TPE and GP-seeded paths.

Consumers: orchestrator.py, sim_batch.py.
"""

import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
STARTING_POINT_FILE = HERE / "derived_starting_point.json"

# Fraction of a screening batch drawn as Gaussian perturbations around the
# starting point (the rest stays with the sampler for exploration). Row 0
# of the injected block is always the exact starting point itself.
EXPLOIT_FRACTION = 0.4
PERTURBATION_STD = 150.0  # volts -- same scale the GP-seeded path uses

# Once a real SIMION trial with hits exists, a second block exploits
# TIGHTLY around it. Rationale (measured, 2026-07-03 campaign): the RK4
# proxy finds the right neighborhood but its fine ranking barely
# correlates with SIMION hits (Pearson +0.106), and hits are rare (0-5 of
# 500 ions) -- so local search around the best CONFIRMED config, at a std
# well below the 150V discovery scale, is the cheapest way to climb.
BEST_FRACTION = 0.3
BEST_PERTURBATION_STD = 40.0  # volts


def load_starting_point(n_electrodes=19):
    """
    Returns the derived starting point as a length-n_electrodes voltage
    array with ONLY the optimized electrodes filled in (fixed electrodes
    are 0 here -- callers overlay FIXED themselves), or None if
    bender_field_analysis.py hasn't produced one yet.
    """
    if not STARTING_POINT_FILE.exists():
        return None
    with open(STARTING_POINT_FILE) as f:
        params = json.load(f)
    v = np.zeros(n_electrodes)
    for key, value in params.items():
        v[int(key.lstrip("V")) - 1] = float(value)
    return v


def inject_starting_point(voltages_batch, optimize, rng=None,
                          exploit_fraction=EXPLOIT_FRACTION,
                          perturbation_std=PERTURBATION_STD,
                          best_center=None, best_fraction=BEST_FRACTION,
                          best_std=BEST_PERTURBATION_STD):
    """
    Overwrite the leading rows of an already-sampled screening batch with
    exploitation blocks:

      rows [0 .. n_best)          tight Gaussian perturbations (best_std)
                                  around best_center, if one was given --
                                  the best CONFIRMED SIMION result;
      rows [n_best .. n_best+n_sp) the derived starting point (first row of
                                  the block exact, rest perturbed with
                                  perturbation_std);
      remaining rows              left as the caller sampled them
                                  (sampler-driven exploration).

    When best_center is given, the starting-point block's fraction is
    halved -- the tight block takes over part of the exploitation budget
    rather than squeezing out the sampler's exploration share.

    Parameters
    ----------
    voltages_batch : (M, n_electrodes) ndarray, modified IN PLACE.
        Fixed electrodes' columns are left untouched -- only the
        `optimize` electrodes are overwritten, so the FIXED overlay the
        caller already applied survives.
    optimize : dict {electrode_number: (low, high)} -- optimizer.py's OPTIMIZE.
    rng : np.random.Generator, optional.
    best_center : length-n_electrodes array, optional
        Full voltage array of the best real SIMION trial (with hits > 0).

    Returns
    -------
    (n_sp, n_best) -- rows written around the starting point / around
    best_center. (0, 0) if neither source is available.
    """
    sp = load_starting_point(voltages_batch.shape[1])
    if sp is None and best_center is None:
        return 0, 0
    rng = rng or np.random.default_rng()
    m = voltages_batch.shape[0]

    n_best = 0
    if best_center is not None:
        n_best = max(1, min(m, int(round(m * best_fraction))))
        exploit_fraction = exploit_fraction / 2.0
    n_sp = 0
    if sp is not None:
        n_sp = max(1, min(m - n_best, int(round(m * exploit_fraction))))

    row = 0
    for _ in range(n_best):
        for e, (low, high) in optimize.items():
            value = best_center[e - 1] + rng.normal(0.0, best_std)
            voltages_batch[row, e - 1] = np.clip(value, low, high)
        row += 1
    for j in range(n_sp):
        for e, (low, high) in optimize.items():
            value = sp[e - 1]
            if j > 0:  # first starting-point row is the exact point
                value += rng.normal(0.0, perturbation_std)
            voltages_batch[row, e - 1] = np.clip(value, low, high)
        row += 1
    return n_sp, n_best
