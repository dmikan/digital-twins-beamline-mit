"""
compare_samplers.py
=====================

Empirical comparison of optimizer.py's current sampler (Optuna's default,
TPESampler -- create_study() there never passes sampler=) against
optimizer_gp.py's explicit GPSampler.

Both samplers are seeded with the IDENTICAL synthetic trial history (same
voltage arrays, same objective values -- a smooth synthetic function
standing in for a real SIMION/RK4 score, so the comparison isn't muddied
by real-run noise or cost) and then asked to suggest the next trial, to
show:
  1. How suggestion wall-clock time scales with trial count for each --
     this is the real structural difference, not suggestion quality.
  2. Where each one proposes to sample next, given the same history.

Run with:
    python compare_samplers.py
"""

import time

import numpy as np
import optuna

import optimizer as op

optuna.logging.set_verbosity(optuna.logging.WARNING)

OPTIMIZE_KEYS = sorted(op.OPTIMIZE)  # electrodes 3,6,9,10,11,12,15,18


def synthetic_objective(params):
    """
    Stand-in for a real SIMION/RK4 score: smooth, deterministic function of
    the 8 free electrodes with one clear optimum plus some ripple, so a
    real surrogate model has SOMETHING structured to learn from (a fair
    comparison needs the trial history to be informative, not just noise).
    """
    x = np.array([params[f"V{k}"] for k in OPTIMIZE_KEYS]) / 1000.0  # normalize to [-1,1]
    target = np.array([0.3, -0.5, 0.2, 0.6, -0.1, 0.4, -0.3, 0.5])
    dist2 = np.sum((x - target) ** 2)
    ripple = 0.1 * np.sum(np.sin(5 * x))
    return -dist2 + ripple  # maximize


def build_seeded_study(sampler, n_seed_trials, seed=0):
    """Fresh in-memory study, fed n_seed_trials IDENTICAL (by construction,
    same seed) completed trials via ask/tell, so both samplers see the same
    history before being timed."""
    study = optuna.create_study(direction="maximize", sampler=sampler)
    rng = np.random.default_rng(seed)
    for _ in range(n_seed_trials):
        trial = study.ask()
        params = {f"V{k}": trial.suggest_float(f"V{k}", -1000.0, 1000.0) for k in OPTIMIZE_KEYS}
        # Overwrite with a fixed pre-drawn point so BOTH studies get the
        # exact same (params, value) pairs regardless of each sampler's own
        # internal random draws for suggest_float during seeding.
        fixed_params = {f"V{k}": float(v) for k, v in zip(OPTIMIZE_KEYS, rng.uniform(-1000, 1000, len(OPTIMIZE_KEYS)))}
        value = synthetic_objective(fixed_params)
        study.tell(trial, value)
        # patch the just-completed trial's params to the fixed ones (ask/tell
        # doesn't let us set suggest_float's return value directly)
        study.trials[-1].params.update(fixed_params)
    return study


def time_suggestion(study, sampler_name):
    t0 = time.time()
    trial = study.ask()
    for k in OPTIMIZE_KEYS:
        trial.suggest_float(f"V{k}", -1000.0, 1000.0)
    elapsed = time.time() - t0
    return elapsed, trial.params


def main():
    print("=== Structural comparison ===")
    print(f"optimizer.py:     sampler = {type(optuna.create_study(direction='maximize').sampler).__name__} "
          f"(Optuna's default -- create_study() there never passes sampler=)")
    print(f"optimizer_gp.py:  sampler = GPSampler (explicit)")
    print()
    print("TPESampler: two KDE density estimators (good trials vs bad trials), no single smooth")
    print("  surrogate, no explicit uncertainty estimate. Cheap per-suggestion regardless of trial count.")
    print("GPSampler:  fits one Gaussian Process (Matern kernel) over all continuous params, WITH an")
    print("  explicit posterior uncertainty (variance) it can be pointed at directly -- this is what makes")
    print("  it fit the project's stated goal of 'explore where the GP is uncertain'. Cost: refitting the")
    print("  GP scales ~cubically with trial count, so suggestions get slower as history grows.")
    print()

    print("=== Timing: suggestion cost vs trial-history size ===")
    print(f"{'n_prior_trials':>15} {'TPE (default)':>16} {'GPSampler':>14}")
    for n in (10, 25, 50, 100, 150):
        tpe_study = build_seeded_study(optuna.samplers.TPESampler(seed=0), n_seed_trials=n, seed=1)
        gp_study = build_seeded_study(optuna.samplers.GPSampler(seed=0), n_seed_trials=n, seed=1)

        t_tpe, _ = time_suggestion(tpe_study, "TPE")
        t_gp, _ = time_suggestion(gp_study, "GP")
        print(f"{n:>15} {t_tpe*1000:>13.1f}ms {t_gp*1000:>13.1f}ms")

    print()
    print("=== Where each one suggests next, given the SAME 50-trial history ===")
    tpe_study = build_seeded_study(optuna.samplers.TPESampler(seed=0), n_seed_trials=50, seed=1)
    gp_study = build_seeded_study(optuna.samplers.GPSampler(seed=0), n_seed_trials=50, seed=1)
    _, tpe_params = time_suggestion(tpe_study, "TPE")
    _, gp_params = time_suggestion(gp_study, "GP")
    for k in OPTIMIZE_KEYS:
        print(f"  V{k:<3d}  TPE: {tpe_params[f'V{k}']:+9.1f}   GP: {gp_params[f'V{k}']:+9.1f}")


if __name__ == "__main__":
    main()
