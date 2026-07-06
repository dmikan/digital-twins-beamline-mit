"""
validate_rk4_filter.py
=======================

Offline validation of the RK4 screening filter against ground truth we
already paid for: every archived study holds (voltage set -> real SIMION
outcome) pairs. This script re-screens those exact voltage sets with the
CURRENT RK4 filter and reports rank correlation against the recorded
SIMION hit counts -- so any filter change can be judged in minutes,
without spending a single new SIMION run.

Scores computed per voltage set (one shared RK4 flight, two scorings):
  old filter -- combined_score aggregation as used before 2026-07-03:
                exp(-d/150) reward - wall_fraction - 0.3*lost_fraction,
                walls flagged but NOT terminating (higher = better).
                Historical baseline: Pearson +0.106 vs hits over 98 pairs.
  new filter -- orchestrator.rk4_score_chunk: wall hits TERMINATE the
                particle, score = best-10% closest-approach distance
                + 20mm * (1 - reached fraction) (lower = better).

Run with:
    python validate_rk4_filter.py
"""

import glob
import pathlib
import sys

import numpy as np
import optuna
from scipy.stats import pearsonr, spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import optimizer as op
from RK4_sim_basis import IonSpecies
from RK4_sim_basis_batch import (
    BatchBasisFieldMap, make_batch_beam, BatchTrajectory, BatchRK4Integrator,
)
from beam_progress_score import make_beam, BeamProgressScorer
from electrode_geometry import build_wall_index
from orchestrator import (
    DETECTOR_BBOX, WALL_EXCLUDE, SCORE_WEIGHTS, SPLAT_TOP_FRACTION,
    HIT_TERM_SCALE_MM, N_ELECTRODES, DT,
)

HERE = ROOT  # datos (CSVs, STLs) en la raiz del proyecto
SCREEN_N = 50          # stage-A fidelity: this is the filter that does the bulk rejection
SCREEN_STEPS = 1500
CHUNK = 50

optuna.logging.set_verbosity(optuna.logging.WARNING)


def collect_archived_trials():
    """(voltages (M,19), mean hits (M,)) from every archived study,
    deduped on the optimized electrodes (0.1V rounding), hits averaged
    across repeat flies of the same voltages."""
    dbs = sorted(glob.glob(str(ROOT / "studies" / "beamline_study_*.db"))) + \
          sorted(glob.glob(str(ROOT / "legacy" / "studies" / "beamline_study_*.db")))
    optimize_keys = sorted(op.OPTIMIZE)
    pooled = {}  # key -> [voltage_row, [hits, hits, ...]]
    n_trials = 0
    for db in dbs:
        try:
            summaries = optuna.study.get_all_study_summaries(storage=f"sqlite:///{db}")
        except Exception as exc:
            print(f"  (skipping {pathlib.Path(db).name}: {exc})")
            continue
        for s in summaries:
            study = optuna.load_study(study_name=s.study_name, storage=f"sqlite:///{db}")
            hits_is_value = study.direction == optuna.study.StudyDirection.MAXIMIZE
            for t in study.trials:
                if t.value is None or not all(f"V{e}" in t.params for e in optimize_keys):
                    continue
                hits = t.user_attrs.get("simion_hits")
                if hits is None:
                    if not hits_is_value or t.value < 0:  # minimize study without attrs, or BAD_SCORE
                        continue
                    hits = t.value
                key = tuple(round(t.params[f"V{e}"], 1) for e in optimize_keys)
                if key not in pooled:
                    v = np.zeros(N_ELECTRODES)
                    for e_num, v_fixed in op.FIXED.items():
                        v[e_num - 1] = v_fixed
                    for e_num in optimize_keys:
                        v[e_num - 1] = t.params[f"V{e_num}"]
                    pooled[key] = [v, []]
                pooled[key][1].append(float(hits))
                n_trials += 1
    voltages = np.array([v for v, _ in pooled.values()])
    hits = np.array([np.mean(h) for _, h in pooled.values()])
    print(f"Collected {n_trials} archived SIMION trials -> {len(pooled)} unique voltage sets "
          f"({int((hits > 0).sum())} with hits > 0)")
    return voltages, hits


# (name, terminate_on_wall_hit, wall_hit_margin, formula, higher_is_better)
# formula "old": exp-reward - wall - 0.3*lost.  "new": best-10% distance
# + 20mm*(1 - reached).  The margin sweep exists because WallIndex
# distances are approximate over-estimates (2mm surface sampling): at
# 1.5mm with termination ON, ~92% of particles die at the einzel and the
# surviving "leading edge" is too small a sample to rank configs by.
# Round 2 (after the termination sweep): termination lost at EVERY margin
# (7-9/61 hitters in top-30 vs 16-17 without -- kill locations are only
# as good as the ~11-15mm STL-transform residuals, and killing erases the
# downstream trajectory info that actually discriminates). So: wall
# proximity as a PENALTY TERM (mm units) on the aligned formula instead.
# formula "new+wallW": best-10% dist + 20mm*(1-reached) + W*wall_fraction.
VARIANTS = [
    ("old  flag        ", False, 1.5, "old", True),
    ("new  noterm      ", False, 1.5, "new", False),
    ("new  +wall*20    ", False, 1.5, "new+wall20", False),
    ("new  +wall*50    ", False, 1.5, "new+wall50", False),
    ("new  +wall*100   ", False, 1.5, "new+wall100", False),
]


def score_variants(bfm, wall_index, voltages, start_positions, start_velocities, species):
    """One RK4 flight per chunk, scored by every VARIANTS entry."""
    M = voltages.shape[0]
    n = start_positions.shape[0]
    n_keep = max(1, int(np.ceil(n * SPLAT_TOP_FRACTION)))
    out = {name: np.empty(M) for name, *_ in VARIANTS}

    for lo in range(0, M, CHUNK):
        hi = min(lo + CHUNK, M)
        chunk = voltages[lo:hi]
        n_cfg = chunk.shape[0]

        bfm.set_voltages_batch(chunk)
        beam, ci = make_batch_beam(species, start_positions, start_velocities, n_cfg)
        trajectory = BatchTrajectory(beam, ci)
        BatchRK4Integrator(bfm, ci).integrate(trajectory, dt=DT, num_steps=SCREEN_STEPS)

        for name, terminate, margin, formula, _ in VARIANTS:
            scorer = BeamProgressScorer(
                bfm=bfm, Trajectory=trajectory, dt=DT, num_steps=SCREEN_STEPS,
                detector_bbox=DETECTOR_BBOX, wall_index=wall_index, wall_hit_margin=margin,
                wall_check_midpoints=False, wall_check_stride=3,
                terminate_on_wall_hit=terminate,
            )
            r = scorer.combined_score(chunk, **SCORE_WEIGHTS)
            if formula.startswith("new"):
                tdist = r["target_distance"].reshape(n_cfg, n)
                reached = r["reached_target"].reshape(n_cfg, n).mean(axis=1)
                top = np.sort(tdist, axis=1)[:, :n_keep].mean(axis=1)
                value = top + HIT_TERM_SCALE_MM * (1.0 - reached)
                if formula.startswith("new+wall"):
                    wall_weight_mm = float(formula.removeprefix("new+wall"))
                    wall = r["hit_wall"].reshape(n_cfg, n).mean(axis=1)
                    value = value + wall_weight_mm * wall
                out[name][lo:hi] = value
            else:
                reward = r["target_reward"].reshape(n_cfg, n).mean(axis=1)
                wall = r["hit_wall"].reshape(n_cfg, n).mean(axis=1)
                lost = r["lost"].reshape(n_cfg, n).mean(axis=1)
                out[name][lo:hi] = (SCORE_WEIGHTS["target_weight"] * reward
                                    - SCORE_WEIGHTS["wall_weight"] * wall
                                    - SCORE_WEIGHTS["lost_weight"] * lost)
        print(f"  scored configs {lo}-{hi - 1}")
    return out


def main():
    voltages, hits = collect_archived_trials()
    if len(voltages) < 10:
        raise SystemExit("Not enough archived trials to validate against.")

    print("Loading field map...")
    bfm = BatchBasisFieldMap.from_directory(HERE, n_electrodes=N_ELECTRODES)
    print("Building wall index...")
    wall_index = build_wall_index(HERE, exclude=WALL_EXCLUDE, target_spacing=2.0)
    species = IonSpecies(mass=28 * 1.66053906660e-27, charge=1.602176634e-19)
    start_positions, start_velocities = make_beam(
        N=SCREEN_N, species=species, start_point=[395.0, 75.0, 77.0],
        mean_energy_eV=15.0, std_energy_eV=0.42466, half_angle_deg=15.0, seed=42)

    print(f"Re-screening {len(voltages)} voltage sets with {len(VARIANTS)} filter variants "
          f"({SCREEN_N} particles x {SCREEN_STEPS} steps)...")
    scores = score_variants(bfm, wall_index, voltages, start_positions, start_velocities, species)

    # Higher-is-better scores correlate positively with hits when good;
    # lower-is-better ones are negated so every row reads the same way.
    hitter_idx = set(np.where(hits > 0)[0])
    print(f"\n===== FILTER VARIANTS vs REAL SIMION HITS "
          f"(n={len(hits)} unique voltage sets, {len(hitter_idx)} hitters) =====")
    print(f"  {'variant':<20} {'Pearson':>8} {'Spearman':>9}   hitters in top-30")
    for name, _, _, _, higher_better in VARIANTS:
        s = scores[name] if higher_better else -scores[name]
        pr, _ = pearsonr(s, hits)
        sr, _ = spearmanr(s, hits)
        top30 = set(int(i) for i in np.argsort(-s)[:30])
        found = len(top30 & hitter_idx)
        print(f"  {name:<20} {pr:>+8.3f} {sr:>+9.3f}   {found}/{len(hitter_idx)}")
    print("  (historical baseline before today's filter work: Pearson +0.106)")


if __name__ == "__main__":
    main()
