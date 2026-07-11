"""
optimizer_gp.py
=================

Same SIMION-driven optimization loop as optimizer.py, but with an explicit
Gaussian-Process sampler (optuna.samplers.GPSampler) instead of Optuna's
default (TPESampler, which optimizer.py gets implicitly -- it never passes
`sampler=` to create_study()). Does NOT modify optimizer.py; reuses its
config/helpers (FIXED, OPTIMIZE, DETECTOR_REGION, get_positions,
count_hits, beam_spread, check_setup) via import, and uses a SEPARATE
study name + database file so this never touches optimizer.py's existing
trial history.

ALSO FIXES A REAL BUG found while measuring SIMION timing for
cost_planner.py: optimizer.py's run_simion() invokes a bare "simion"
command via subprocess(shell=True). In this environment that command is
NOT resolvable (cmd.exe reports "'simion' is not recognized") even with
cwd=SIMION_INSTALL_DIR -- it only works with the fully-qualified path to
simion.exe. Fixed here; worth porting back to optimizer.py separately.

WHY A GP SAMPLER FITS THIS PROJECT
-------------------------------------
Session context: the RK4 batch pipeline (sim_batch.py) exists specifically
to cheaply explore/rank voltage space BEFORE spending SIMION runs, and the
stated goal was "investigate spaces of zero gradient in the GP space to
improve the ability to reach the goal". GPSampler is the sampler that
actually HAS an explicit notion of "the GP's uncertainty" (the posterior
variance) to investigate -- TPE (the current default) doesn't fit a single
smooth surrogate with quantified uncertainty at all; see
compare_samplers.py for what that difference looks like in practice.

Run with:
    python optimizer_gp.py
"""

import pathlib
import subprocess

import optuna
from optuna import Trial

from optimizer import (
    FIXED, OPTIMIZE, DETECTOR_REGION, N_TRIALS, STARTING_POINT,
    OBJECTIVE, BAD_SCORE, DIRECTION,
    SIM_FOLDER, IOB_FILE, PA0_FILE, SIMION_INSTALL_DIR,
    get_positions, count_hits, beam_spread, check_setup,
)

# Separate from optimizer.py's beamline_study.db / "minimalsim_practice" --
# this sampler's trial history should never mix with the TPE study's.
RESULTS_DB = SIM_FOLDER / "beamline_study_gp.db"
RESULTS_CSV = SIM_FOLDER / "beamline_results_gp.csv"
STUDY_NAME = "minimalsim_practice_gp"

# Bug fix vs optimizer.py: bare "simion" isn't resolvable via
# subprocess(shell=True) in this environment -- needs the qualified path.
SIMION_EXE = SIMION_INSTALL_DIR / "simion.exe"

FLY_COMMAND = (
    f'"{SIMION_EXE}" --nogui fly --recording-output=out.txt --programs=0 '
    f'--retain-trajectories=0 --restore-potential=0 "{IOB_FILE}"'
)


def run_simion(command: str) -> str:
    result = subprocess.run(
        command, cwd=str(SIMION_INSTALL_DIR), shell=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
    )
    return result.stdout


def apply_voltages(chosen: dict) -> None:
    all_volts = {**FIXED, **chosen}
    settings = ",".join(f"{n}={v}" for n, v in sorted(all_volts.items()))
    run_simion(f'"{SIMION_EXE}" --nogui fastadj "{PA0_FILE}" {settings}')


def objective(trial: Trial) -> float:
    chosen = {}
    for number, (low, high) in OPTIMIZE.items():
        chosen[number] = trial.suggest_float(f"V{number}", low, high)

    try:
        apply_voltages(chosen)
        simion_output = run_simion(FLY_COMMAND)
    except subprocess.CalledProcessError:
        return BAD_SCORE

    positions = get_positions(simion_output)
    if positions.shape[0] == 0:
        return BAD_SCORE

    return count_hits(positions) if OBJECTIVE == "hits" else beam_spread(positions)


def main() -> None:
    print("Beamline voltage optimizer -- GPSampler version")
    print(f"Goal: {'MAXIMIZE ions on detector' if OBJECTIVE == 'hits' else 'MINIMIZE beam spread'}")
    print(f"Optimizing electrodes: {sorted(OPTIMIZE)}")
    print(f"Sampler: GPSampler (vs optimizer.py's default TPESampler)")

    check_setup()

    study = optuna.create_study(
        direction=DIRECTION,
        storage=f"sqlite:///{RESULTS_DB}",
        study_name=STUDY_NAME,
        load_if_exists=True,
        sampler=optuna.samplers.GPSampler(seed=42),
    )
    if STARTING_POINT:
        study.enqueue_trial(STARTING_POINT)
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=True)

    print("Optimization completed.")

    try:
        study.trials_dataframe().to_csv(RESULTS_CSV, index=False)
        print(f"Saved all results to {RESULTS_CSV}")
    except Exception as error:
        print(f"Could not save the results CSV: {error}")

    best = study.best_trial
    print("\n=== Best trial ===")
    print(f"Trial number: {best.number}")
    print(f"Score:        {best.value}")
    print("Voltages:")
    for name, value in best.params.items():
        print(f"   {name}: {value}")


if __name__ == "__main__":
    main()
