"""
report_orchestrator_run.py
=============================

Reads orchestrator.py's persisted study (beamline_study_orchestrator.db)
and writes a human-readable .txt report: per-trial breakdown (voltages,
real SIMION result, the RK4 score that got it promoted, which run/
iteration it came from), plus summary stats -- including whether the RK4
ranking actually correlates with the real SIMION outcome, which is the
whole point of screening with it in the first place.

Run with:
    python report_orchestrator_run.py [output_path.txt]
"""

import sys
import pathlib
from datetime import datetime

import numpy as np
import optuna

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orchestrator import STUDY_NAME, RESULTS_DB

optuna.logging.set_verbosity(optuna.logging.WARNING)


def generate_report(out_path):
    study = optuna.load_study(study_name=STUDY_NAME, storage=f"sqlite:///{RESULTS_DB}")
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    lines = []
    w = lines.append

    w("=" * 78)
    w("ORCHESTRATOR RUN REPORT")
    w(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    w(f"Study: {STUDY_NAME}  ({RESULTS_DB.name})")
    w("=" * 78)
    w("")
    w(f"Total real SIMION trials on record: {len(trials)}")
    w(f"Objective: maximize hits on detector (see optimizer.py OBJECTIVE)")
    w("")

    values = np.array([t.value for t in trials])
    w(f"Best value found:  {values.max():.1f}")
    w(f"Worst value found: {values.min():.1f}")
    w(f"Mean value:        {values.mean():.2f}")
    w(f"Trials with hits > 0: {(values > 0).sum()} / {len(trials)}")
    w("")

    # --- Per-trial table ---
    w("-" * 78)
    w("PER-TRIAL DETAIL (in trial-number order -- includes ALL runs ever")
    w("recorded on this study, not just the most recent one)")
    w("-" * 78)
    header = f"{'#':>4} {'iter':>5} {'rk4rank':>8} {'rk4_score':>10} {'SIMION':>8} {'SIMION_s':>9}  voltages (V3,V6,V9,V10,V11,V12,V15,V18)"
    w(header)
    w("-" * len(header))
    for t in trials:
        ua = t.user_attrs
        it = ua.get("iteration", "-")
        rank = ua.get("rk4_rank_in_iteration", "-")
        rk4s = ua.get("rk4_score", None)
        rk4s_str = f"{rk4s:+.3f}" if rk4s is not None else "n/a"
        simion_s = ua.get("simion_elapsed_s", None)
        simion_s_str = f"{simion_s:.1f}" if simion_s is not None else "n/a"
        volts = ",".join(f"{t.params[f'V{k}']:.0f}" for k in (3, 6, 9, 10, 11, 12, 15, 18))
        w(f"{t.number:>4} {it!s:>5} {rank!s:>8} {rk4s_str:>10} {t.value:>8.1f} {simion_s_str:>9}  {volts}")
    w("")

    # --- RK4 vs real correlation ---
    w("-" * 78)
    w("DOES THE RK4 SCORE PREDICT THE REAL SIMION OUTCOME?")
    w("-" * 78)
    with_rk4 = [t for t in trials if "rk4_score" in t.user_attrs]
    if len(with_rk4) >= 3:
        rk4_scores = np.array([t.user_attrs["rk4_score"] for t in with_rk4])
        real_values = np.array([t.value for t in with_rk4])
        if np.std(rk4_scores) > 1e-9 and np.std(real_values) > 1e-9:
            corr = np.corrcoef(rk4_scores, real_values)[0, 1]
            w(f"Pearson correlation (RK4 combined_score vs real SIMION hits), n={len(with_rk4)}: {corr:+.3f}")
        else:
            w(f"Can't compute a meaningful correlation -- one of the two series is constant "
              f"(rk4_score std={np.std(rk4_scores):.4f}, real_value std={np.std(real_values):.4f}).")
            w("This is expected if every promoted candidate scored the same on one or both")
            w("axes (e.g. all real SIMION runs returned 0 hits, or all RK4 scores tied because")
            w("no promoted candidate got meaningfully closer to the detector than any other --")
            w("see this project's earlier finding that random voltage sampling essentially")
            w("never reaches the detector region at all).")
    else:
        w("Not enough trials with recorded rk4_score to compute a correlation.")
    w("")

    # --- Per-iteration summary (most recent run only, if iteration info present) ---
    w("-" * 78)
    w("PER-ITERATION SUMMARY (most recent orchestrate() call)")
    w("-" * 78)
    iters = sorted({t.user_attrs.get("iteration") for t in with_rk4 if t.user_attrs.get("iteration") is not None})
    for it in iters:
        it_trials = [t for t in with_rk4 if t.user_attrs.get("iteration") == it]
        it_values = [t.value for t in it_trials]
        w(f"  iteration {it}: {len(it_trials)} SIMION runs, values={it_values}, "
          f"best={max(it_values):.1f}")
    w("")
    w("NOTE: 'iteration' numbers reset to 1 at the start of every orchestrate() call --")
    w("if this study has been run more than once, trials from different runs may share")
    w("the same iteration number. Use trial # (globally unique) to distinguish runs.")
    w("")

    # --- Best trial detail ---
    best = study.best_trial
    w("-" * 78)
    w("BEST TRIAL")
    w("-" * 78)
    w(f"Trial #{best.number}, value={best.value}")
    for name, value in best.params.items():
        w(f"   {name}: {value:.2f}")
    w("")
    w("=" * 78)

    report = "\n".join(lines)
    out_path.write_text(report, encoding="utf-8")
    print(f"Report written to {out_path}")
    return report


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "orchestrator_report.txt"
    generate_report(out)
