"""
fresh_run.py
=============

Clean-slate benchmark of the current pipeline: runs the exact same closed
loop as orchestrator.py (RK4 screen -> SIMION confirm -> feed back) but
against a BRAND NEW Optuna study with zero prior trials, to measure how
the system performs starting from nothing.

What "from scratch" means here:
  - New study name + new sqlite DB (below) -- none of the 130 trials
    accumulated in legacy/studies/beamline_study_orchestrator.db inform
    the sampler.
  - The physics-informed starting point (derived_starting_point.json) IS
    still injected: it comes from field analysis, not from any study
    history, so it's part of the method being benchmarked. The tight
    exploit-around-best block only activates once THIS study finds its
    own first hit.

Usage:
    python fresh_run.py [budget] [--continue]

Defaults to budget 50 (5 iterations x 10 SIMION runs). Without
--continue the script REFUSES to run if the fresh DB already exists, so a
"fresh" result can never silently include earlier trials. --continue
appends to the existing fresh study -- used only to split one long
campaign into shorter processes; it's the same study either way.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import orchestrator

FRESH_STUDY = "fresh_from_scratch_50"


def main():
    args = list(sys.argv[1:])
    cont = "--continue" in args
    args = [a for a in args if a != "--continue"]
    budget = int(args[0]) if args else 50
    # "_v2": el objetivo cambio a J_v2 normalizado [0,1] (2026-07-05) --
    # los estudios _splat (escala mm) y los de hits NUNCA deben
    # continuarse con este codigo.
    FRESH_DB = ROOT / "studies" / f"beamline_study_fresh{budget}_v2.db"

    if FRESH_DB.exists() and not cont:
        raise SystemExit(
            f"{FRESH_DB.name} already exists -- this run would NOT be from "
            f"scratch. Move/delete it (or pass --continue to append to it)."
        )

    # orchestrate()/build_study() read these module globals -- point them
    # at the fresh study instead of the historical one.
    orchestrator.STUDY_NAME = FRESH_STUDY
    orchestrator.RESULTS_DB = FRESH_DB

    study = orchestrator.orchestrate(total_simion_budget=budget, simion_per_iteration=10)

    trials = [t for t in study.trials if t.value is not None]
    # Objective is now mean splat distance (minimize); hit counts live in
    # user_attrs["simion_hits"]. Show the 10 closest-approach trials.
    hitters = [t for t in trials if t.user_attrs.get("simion_hits", 0) > 0]
    closest = sorted(trials, key=lambda t: t.value)[:10]
    print(f"\n===== FRESH-STUDY SUMMARY ({FRESH_STUDY}) =====")
    print(f"SIMION trials in this study: {len(trials)}   with hits: {len(hitters)}")
    print("10 best by mean splat distance:")
    for t in closest:
        volts = "  ".join(f"{k}={v:+.0f}" for k, v in
                          sorted(t.params.items(), key=lambda kv: int(kv[0][1:])))
        print(f"  trial {t.number:3d}: dist={t.value:7.1f}mm  "
              f"hits={t.user_attrs.get('simion_hits', 0):3d}  {volts}")


if __name__ == "__main__":
    main()
