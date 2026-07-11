"""
cost_planner.py
=================

Time-cost model for choosing (a) the RK4 screening fidelity (particles,
steps), (b) how many candidates M to RK4-screen, and (c) what
percentage/count K of those to promote to real SIMION runs -- derived from
REAL measured per-call costs on this project/machine, not guessed.

MEASURED SIMION COST
----------------------
fastadj + fly, per candidate: ~5.77s (5.04s fastadj -- dominated by
loading/saving the 289MB electrode_.PA0 file and SIMION's own process-
startup cost EACH invocation, not the actual computation -- plus 0.73s
fly). Note: optimizer.py's run_simion() invokes a bare "simion" command
that isn't resolvable via subprocess(shell=True) in this environment (it
needs the fully-qualified path to simion.exe) -- worth fixing there
separately; this module's SIMION_COST_PER_CANDIDATE_S was measured with
the qualified path.

MEASURED RK4 COST
-------------------
IMPORTANT: this was recalibrated once -- the first version was fit on the
single-config RK4Integrator/BasisFieldMap.field() pathway (3-D
interpolation), not the batched BatchRK4Integrator/field_batch() pathway
(4-D interpolation, with config as an extra axis) that sim_batch.py
actually uses. Verified in production: sim_batch.py predicted 115.4s for
M=491 and actually took 250.1s (2.17x off) -- because field_batch()'s 4-D
interpolation genuinely costs more per particle than field()'s 3-D case,
not just batching overhead. Recalibrated directly against
BatchRK4Integrator (5 points varying M at fixed N,T and T at fixed M,N):

    cost(total_particles = M*N, T steps) ~= T * (a + b*total_particles)  seconds

    a = 1.80e-4 s/step        -- FIXED overhead per step, paid once per
                                  batch regardless of M (Python loop, state
                                  allocation).
    b = 2.80e-6 s/particle-step -- marginal 4-D interpolation cost. This is
                                  ~1.6x the single-config 3-D interpolation
                                  cost (~1.71e-6) -- real, not noise.

    => cost_batch(M, N, T) ~= T*a + T*b*N*M   (same form, corrected constants)

Practical consequence unchanged: cutting T (steps) is still far more
effective than cutting N (particles) for the same total speedup, since T
controls how many times the fixed per-step cost is paid.

SCREENING FIDELITY: T has a physics floor, not just a speed target
------------------------------------------------------------------------
RECALIBRATED (2026-07-03) after the RK4 acceleration unit fix
(RK4_sim_basis_batch.E_VMM_TO_ACCEL_MM) and the DT change 5e-8 -> 1e-8:
the original floor here ("T<1000 useless, 0% resolved at T=500,
score drifts to T=12000") was measured on the BROKEN dynamics, where the
beam was never accelerated off the +500V source and crawled at 15 eV.
With corrected physics the beam is ~470 eV and, measured on the derived
starting-point config (200 particles, DT=1e-8, thorough wall check):

    resolved (reached/lost/wall) by step  500:  62%
    resolved by step 1000:  92%     p95 resolve step: 1102
    resolved by step 1500: 100%     p99 resolve step: 1290

The binding constraint is now geometric: REACHING the detector takes
~1300-1400 steps at DT=1e-8 (~650mm of path at ~0.5mm/step), so any T
below that makes a detector hit physically unrecordable and destroys the
target_reward signal for exactly the candidates screening exists to find.
Hence MIN_VIABLE_STEPS=1300 (detector physically reachable) and
RECOMMENDED_SCREENING_STEPS=1500 (100% resolution with margin). These are
DT=1e-8 steps -- rescale both if DT ever changes again.

Particle count for screening (default 50, vs this project's production
500): fewer particles means noisier fraction estimates (wall_hit_fraction,
target_reward mean, lost_fraction all become less precise), but since
screening only needs to RANK candidates well enough to pick a promising
top-K subset -- not report a precise final number -- 50 is a reasonable
trade. If screening ever seems to be promoting bad candidates, raising
this is the first knob to try before raising rk4_steps.
"""

import numpy as np

# ----------------------------------------------------------------------
# Real measured/fit constants
# ----------------------------------------------------------------------
SIMION_COST_PER_CANDIDATE_S = 5.77
RK4_FIXED_COST_PER_STEP_S = 1.80e-4
RK4_MARGINAL_COST_PER_PARTICLE_STEP_S = 2.80e-6

# First version of this model only accounted for RK4Integrator.integrate()
# and underpredicted a real sim_batch.py run by 2.3x (predicted 115s,
# measured 266s for M=535 at the recommended fidelity). Root cause,
# profiled directly on one 50-config chunk at the recommended fidelity:
# combined_score() -- specifically BeamProgressScorer._wall_hit_mask's
# wall_index.distance() calls -- was ~86% of scoring time, and scoring
# added ~37% on top of pure integration time (8.87s scoring vs 24.17s
# integration for that chunk).
#
# Fixed properly, not just re-measured: BeamProgressScorer gained
# wall_check_midpoints/wall_check_stride (see that file) to control how
# many points the wall check queries. Tested wall_check_midpoints=False,
# wall_check_stride=3 against the thorough default across 3 different
# random voltage batches -- IDENTICAL wall_hit_fraction/combined_score
# every time, at ~4.2x less scoring time (8.6s -> ~2.1s for a 50-config
# chunk). stride=5 was also tried and DID start losing signal (wall_hit
# fraction shifted), so 3 is a validated floor, not a guess. sim_batch.py
# uses these settings by default now. Multiplier measured directly against
# the corrected RK4_FIXED/MARGINAL constants above (integrate 15.4s +
# score 2.2s for a 50-config chunk): 17.6/15.4 ~= 1.14.
SCORING_OVERHEAD_MULTIPLIER = 1.14

MIN_VIABLE_STEPS = 1300               # below this the detector is physically unreachable at DT=1e-8
RECOMMENDED_SCREENING_STEPS = 1500    # 100% of particles resolved by here -- see module docstring
RECOMMENDED_SCREENING_PARTICLES = 50  # enough for a usable fraction estimate, see note above


def rk4_batch_cost(M, N=RECOMMENDED_SCREENING_PARTICLES, T=RECOMMENDED_SCREENING_STEPS,
                    a=RK4_FIXED_COST_PER_STEP_S, b=RK4_MARGINAL_COST_PER_PARTICLE_STEP_S,
                    include_scoring=True):
    """
    Predicted wall-clock seconds to RK4-screen M configs x N particles x T
    steps together AND score them (combined_score) -- include_scoring=False
    gives just the raw integration estimate, e.g. for comparing against the
    measured integration-only numbers in the module docstring.
    """
    integration = T * a + T * b * N * M
    return integration * SCORING_OVERHEAD_MULTIPLIER if include_scoring else integration


def plan_batch(simion_budget_k, rk4_time_budget_s=None, total_time_budget_s=None,
               rk4_particles=RECOMMENDED_SCREENING_PARTICLES,
               rk4_steps=RECOMMENDED_SCREENING_STEPS,
               simion_cost=SIMION_COST_PER_CANDIDATE_S,
               min_subset_fraction=0.01, max_subset_fraction=0.5):
    """
    Choose M (how many candidates to RK4-screen) given a SIMION budget K
    (how many of them you can afford to actually confirm in SIMION).

    Parameters
    ----------
    simion_budget_k : int
        How many candidates you're willing/able to run through real SIMION
        this iteration -- this is a resource decision (SIMION time, ~5.77s
        each), not something derivable from RK4 cost alone. You set this;
        the function figures out how much RK4 screening it buys you.
    rk4_time_budget_s : float, optional
        Wall-clock seconds to spend RK4-screening BEFORE spending
        simion_budget_k * simion_cost on confirmations. Default: equal to
        the SIMION time itself ("screening costs no more than confirming")
        -- since RK4 is ~40x cheaper per candidate at the recommended
        fidelity, an equal time budget already screens ~40x more
        candidates than get promoted.
    total_time_budget_s : float, optional
        Alternative to rk4_time_budget_s: a TOTAL budget for
        screening+confirmation combined; simion_budget_k * simion_cost is
        carved out first, the remainder goes to screening.
    rk4_particles, rk4_steps : int
        Screening fidelity. Raising rk4_steps below MIN_VIABLE_STEPS is
        refused (screening would see 0% resolved particles).
    min_subset_fraction, max_subset_fraction : float
        Sanity bounds on K/M regardless of what the time budget implies
        (e.g. never promote more than 50% of what was screened, never
        bother screening more than 100x what gets promoted).

    Returns
    -------
    dict: rk4_particles, rk4_steps, M, simion_budget_k, subset_fraction,
    predicted_screening_time_s, predicted_simion_time_s, predicted_total_time_s.
    """
    if rk4_steps < MIN_VIABLE_STEPS:
        raise ValueError(
            f"rk4_steps={rk4_steps} is below the measured viability floor "
            f"({MIN_VIABLE_STEPS}) -- screening would see ~0% resolved "
            f"particles and every candidate would look identical."
        )

    simion_time = simion_budget_k * simion_cost

    if rk4_time_budget_s is None:
        if total_time_budget_s is not None:
            rk4_time_budget_s = max(0.0, total_time_budget_s - simion_time)
        else:
            rk4_time_budget_s = simion_time

    # Both terms scaled by SCORING_OVERHEAD_MULTIPLIER so M reflects the
    # REAL achievable count (integration + scoring) within the budget, not
    # just integration -- see that constant's docstring for why.
    fixed = rk4_steps * RK4_FIXED_COST_PER_STEP_S * SCORING_OVERHEAD_MULTIPLIER
    marginal_per_candidate = (rk4_steps * RK4_MARGINAL_COST_PER_PARTICLE_STEP_S
                               * rk4_particles * SCORING_OVERHEAD_MULTIPLIER)

    if rk4_time_budget_s <= fixed:
        raise ValueError(
            f"rk4_time_budget_s={rk4_time_budget_s:.2f}s doesn't even cover the fixed "
            f"per-batch overhead ({fixed:.2f}s at {rk4_steps} steps, including the "
            f"scoring-overhead correction) -- give a bigger budget, fewer steps, or a "
            f"bigger simion_budget_k."
        )

    M = int((rk4_time_budget_s - fixed) / marginal_per_candidate)
    M = max(M, simion_budget_k)  # never screen fewer than we intend to promote

    if simion_budget_k / M > max_subset_fraction:
        M = int(np.ceil(simion_budget_k / max_subset_fraction))
    elif simion_budget_k / M < min_subset_fraction:
        M = int(np.floor(simion_budget_k / min_subset_fraction))

    predicted_screening_time = rk4_batch_cost(M, rk4_particles, rk4_steps)  # includes scoring

    return {
        "rk4_particles": rk4_particles,
        "rk4_steps": rk4_steps,
        "M": M,
        "simion_budget_k": simion_budget_k,
        "subset_fraction": simion_budget_k / M,
        "predicted_screening_time_s": predicted_screening_time,
        "predicted_simion_time_s": simion_time,
        "predicted_total_time_s": predicted_screening_time + simion_time,
    }


if __name__ == "__main__":
    print(f"{'K (SIMION runs)':>18} {'M (RK4 screened)':>18} {'subset %':>10} "
          f"{'RK4 time':>10} {'SIMION time':>12} {'total':>10}")
    for k in (5, 10, 20, 50, 100):
        plan = plan_batch(simion_budget_k=k)
        print(f"{k:>18} {plan['M']:>18} {plan['subset_fraction']*100:>9.2f}% "
              f"{plan['predicted_screening_time_s']:>9.1f}s {plan['predicted_simion_time_s']:>11.1f}s "
              f"{plan['predicted_total_time_s']:>9.1f}s")
