"""
bender_field_analysis.py
=========================

The shared analysis behind follow-up options #2 and #3 (see
SESSION_HANDOFF.txt section 8): sample each quadrupole-bender electrode's
(9, 10, 11, 12) basis field to determine what deflection each one actually
produces per volt, derive the correlated sign PATTERN that turns the beam
from its initial -x heading into the +z heading the detector requires
(~328mm of net z travel from z=77 to z=405), then find the pattern
STRENGTH empirically with a batched-RK4 scan.

Stages (all cheap, no SIMION involved):
  1. Locate electrodes 9-12 in workbench space from their basis potentials
     (the region where basis_V ~= its max is the conductor surface).
  2. Probe each bender electrode's PER-VOLT field along the nominal beam
     path (leg 1: -x from the source at (395,75,77) toward the bender;
     leg 2: +z from the bender toward the detector at ~(76,76,405)) and
     integrate the per-volt deflection impulse each contributes.
  3. Derive the quadrupole pattern: diagonal (opposite) electrodes share a
     polarity, adjacent ones oppose -- pairing read off the located
     geometry, not assumed.
  4. Coarse batched-RK4 scan over pattern strength s (V9..V12 = s *
     pattern), plus single-electrode probe configs for comparison.
  5. Refine around the best s, and sweep the einzel-lens centers (V3, V6)
     at that s.

Outputs:
  - console report (stages 1-5)
  - bender_analysis_results.txt   (same report, persisted)
  - bender_pattern_scan.png       (score / miss distance vs s)
  - derived_starting_point.json   {"V3": ..., ..., "V18": ...} -- the
    physically-informed STARTING_POINT for orchestrator.py (option #3)

Run with:
    python bender_field_analysis.py
"""

import json
import pathlib
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import optimizer as op
from RK4_sim_basis import IonSpecies
from RK4_sim_basis_batch import (
    BatchBasisFieldMap, make_batch_beam, BatchTrajectory, BatchRK4Integrator,
)
from beam_progress_score import make_beam, BeamProgressScorer
from electrode_geometry import build_wall_index

HERE = ROOT       # datos (CSVs, STLs) y derived_starting_point.json en la raiz
OUTPUTS = ROOT / "outputs"
N_ELECTRODES = 19
# dt=1e-8, not the pipeline's historical 5e-8: with the corrected
# acceleration units (see RK4_sim_basis_batch.E_VMM_TO_ACCEL_MM) the beam
# leaves the +500V source at ~470 eV (~5.7e7 mm/s) -- at dt=5e-8 that's
# 2.9mm per step, only ~15 steps across the whole 44mm bender gap. 1e-8
# gives ~0.6mm/step; 2500 steps then covers ~2.5x the full transit time.
DT = 1e-8
BENDER = (9, 10, 11, 12)
EINZEL = (3, 6)
PLATES = (15, 18)

DETECTOR_BBOX = (69.0, 81.0, 68.5, 81.5, 403.0, 407.0)
DETECTOR_CENTER = np.array([75.0, 75.0, 405.0])
WALL_EXCLUDE = (1, 2, 19)
SCORE_WEIGHTS = dict(target_weight=1.0, wall_weight=1.0, lost_weight=0.3, target_scale=150.0)
START_POINT = np.array([395.0, 75.0, 77.0])

# Screening fidelity: 50 particles is the cost-planner recommendation.
# 2500 steps at DT=1e-8 is ~2.5x the full ~650mm transit (320mm of -x plus
# 330mm of +z) at the corrected ~470 eV beam energy. The report prints
# each config's unresolved fraction so it's visible if this is too short.
BEAM_N = 50
NUM_STEPS = 2500
CHUNK = 25

REPORT_LINES = []


def log(msg=""):
    print(msg)
    REPORT_LINES.append(str(msg))


# ----------------------------------------------------------------------
# Stage 1: locate the bender electrodes from their basis potentials
# ----------------------------------------------------------------------
def locate_electrodes(bfm, electrodes):
    """
    Centroid + extent of each electrode's conductor region in workbench
    space, taken as the grid points where its basis potential is >= 99% of
    its own max (the basis field is the solution for THAT electrode at a
    reference potential with all others grounded, so the plateau at max is
    the conductor itself).
    """
    centroids = {}
    log("--- Stage 1: electrode locations (from basis potentials) ---")
    for e in electrodes:
        basis = bfm.basis_V[e - 1]
        vmax = basis.max()
        mask = basis >= 0.99 * vmax
        pts = bfm.coords[mask]
        centroid = pts.mean(axis=0)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        centroids[e] = centroid
        log(f"  electrode {e:2d}: basis max={vmax:.3f}  centroid=({centroid[0]:6.1f}, "
            f"{centroid[1]:6.1f}, {centroid[2]:6.1f})  extent x[{lo[0]:.0f},{hi[0]:.0f}] "
            f"y[{lo[1]:.0f},{hi[1]:.0f}] z[{lo[2]:.0f},{hi[2]:.0f}]")
    return centroids


# ----------------------------------------------------------------------
# Stage 2: per-volt field probe along the nominal beam path
# ----------------------------------------------------------------------
def probe_per_volt_fields(bfm, electrodes, bender_center):
    """
    For each electrode, set 1V on it alone (basis recombination makes this
    exactly its per-volt field) and integrate E along the two legs of the
    nominal path. The integral of Ez over leg 1 is the per-volt z-impulse
    the electrode gives the beam BEFORE/WHILE it turns -- its sign says
    which polarity pushes the beam toward the detector (+z, for a positive
    ion F = qE with q > 0).
    """
    n_samples = 200
    leg1 = np.linspace([START_POINT[0], START_POINT[1], START_POINT[2]],
                       [bender_center[0], START_POINT[1], START_POINT[2]], n_samples)
    leg2 = np.linspace([DETECTOR_CENTER[0], DETECTOR_CENTER[1], bender_center[2]],
                       DETECTOR_CENTER, n_samples)
    ds1 = np.linalg.norm(leg1[1] - leg1[0])
    ds2 = np.linalg.norm(leg2[1] - leg2[0])

    log("\n--- Stage 2: per-volt deflection impulse along the nominal path ---")
    log(f"  leg 1: ({leg1[0][0]:.0f},{leg1[0][1]:.0f},{leg1[0][2]:.0f}) -> "
        f"({leg1[-1][0]:.0f},{leg1[-1][1]:.0f},{leg1[-1][2]:.0f})   "
        f"leg 2: ({leg2[0][0]:.0f},{leg2[0][1]:.0f},{leg2[0][2]:.0f}) -> "
        f"({leg2[-1][0]:.0f},{leg2[-1][1]:.0f},{leg2[-1][2]:.0f})")
    log(f"  {'elec':>4} | {'leg1 int(Ez)ds':>14} {'leg1 int(Ex)ds':>14} {'leg1 int(Ey)ds':>14} "
        f"| {'leg2 int(Ex)ds':>14} {'leg2 int(Ey)ds':>14} {'leg2 int(Ez)ds':>14}   (V*mm/mm per volt)")

    impulses = {}
    for e in electrodes:
        unit = np.zeros(N_ELECTRODES)
        unit[e - 1] = 1.0
        bfm.set_voltages(unit)
        E1 = bfm.field(leg1)   # (n_samples, 3), per volt
        E2 = bfm.field(leg2)
        i1 = E1.sum(axis=0) * ds1   # integral of E ds over leg 1, per volt
        i2 = E2.sum(axis=0) * ds2
        impulses[e] = (i1, i2)
        log(f"  {e:>4} | {i1[2]:>14.4f} {i1[0]:>14.4f} {i1[1]:>14.4f} "
            f"| {i2[0]:>14.4f} {i2[1]:>14.4f} {i2[2]:>14.4f}")
    # Restore all-zero so no probe voltage leaks into later stages.
    bfm.set_voltages(np.zeros(N_ELECTRODES))
    return impulses


# ----------------------------------------------------------------------
# Stage 3: derive the quadrupole pairing from geometry + impulse signs
# ----------------------------------------------------------------------
def derive_pattern(centroids, impulses):
    """
    Quadrupole bender: diagonal (opposite-across-the-axis) electrodes share
    a polarity, adjacent ones oppose. Pairing is read off the located
    centroids: the diagonal partner of an electrode is the one whose offset
    from the bender center points most nearly OPPOSITE its own.

    Sign convention: +1 goes to the pair whose members have a POSITIVE
    leg-1 z-impulse (they push a positive ion toward +z / the detector when
    positive), so a positive scan strength s means "bend toward detector".
    """
    center = np.mean([centroids[e] for e in BENDER], axis=0)
    offsets = {e: centroids[e] - center for e in BENDER}

    e0 = BENDER[0]
    others = [e for e in BENDER if e != e0]
    dots = {e: float(np.dot(offsets[e0], offsets[e])) for e in others}
    partner = min(dots, key=dots.get)  # most-opposite offset = diagonal partner
    pair_a = (e0, partner)
    pair_b = tuple(e for e in others if e != partner)

    log("\n--- Stage 3: quadrupole pairing (from geometry) ---")
    log(f"  bender center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
    for e in BENDER:
        o = offsets[e]
        log(f"  electrode {e:2d} offset from center: ({o[0]:+7.1f}, {o[1]:+7.1f}, {o[2]:+7.1f})")
    log(f"  diagonal pairs: {pair_a} and {pair_b}")

    # Which pair pushes +z (per volt, per leg-1 impulse)? That pair gets +1.
    za = np.mean([impulses[e][0][2] for e in pair_a])
    zb = np.mean([impulses[e][0][2] for e in pair_b])
    pattern = np.zeros(N_ELECTRODES)
    plus_pair, minus_pair = (pair_a, pair_b) if za >= zb else (pair_b, pair_a)
    for e in plus_pair:
        pattern[e - 1] = +1.0
    for e in minus_pair:
        pattern[e - 1] = -1.0
    log(f"  mean leg-1 z-impulse per volt: {pair_a}={za:+.4f}, {pair_b}={zb:+.4f}")
    log(f"  => pattern (+1 pair pushes +z when positive): +{plus_pair}, -{minus_pair}")
    return pattern, center


# ----------------------------------------------------------------------
# Stages 4/5: batched RK4 flights
# ----------------------------------------------------------------------
def base_voltages(m):
    batch = np.zeros((m, N_ELECTRODES))
    for k, v in op.FIXED.items():
        batch[:, k - 1] = v
    return batch


def fly_batch(bfm, wall_index, voltages_batch, start_positions, start_velocities, species):
    """Chunked batched-RK4 flight + scoring. Returns per-config diagnostics."""
    M = voltages_batch.shape[0]
    n = start_positions.shape[0]
    out = {k: np.empty(M) for k in
           ("combined", "tdist_mean", "tdist_min", "wall_frac", "lost_frac",
            "hit_frac", "unresolved_frac", "final_x", "final_y", "final_z")}

    n_chunks = int(np.ceil(M / CHUNK))
    for c in range(n_chunks):
        lo, hi = c * CHUNK, min((c + 1) * CHUNK, M)
        chunk = voltages_batch[lo:hi]
        n_cfg = chunk.shape[0]

        bfm.set_voltages_batch(chunk)
        beam, config_indices = make_batch_beam(species, start_positions, start_velocities, n_cfg)
        trajectory = BatchTrajectory(beam, config_indices)
        BatchRK4Integrator(bfm, config_indices).integrate(trajectory, dt=DT, num_steps=NUM_STEPS)

        scorer = BeamProgressScorer(
            bfm=bfm, Trajectory=trajectory, dt=DT, num_steps=NUM_STEPS,
            detector_bbox=DETECTOR_BBOX, wall_index=wall_index, wall_hit_margin=1.5,
            wall_check_midpoints=False, wall_check_stride=3,
        )
        result = scorer.combined_score(chunk, **SCORE_WEIGHTS)

        tdist = result["target_distance"].reshape(n_cfg, n)
        treward = result["target_reward"].reshape(n_cfg, n)
        wall = result["hit_wall"].reshape(n_cfg, n)
        lost = result["lost"].reshape(n_cfg, n)
        reached = result["reached_target"].reshape(n_cfg, n)

        positions = result["positions"]          # (T, n_cfg*n, 3)
        stop_idx = result["stop_idx"]            # (n_cfg*n,)
        final = positions[stop_idx, np.arange(positions.shape[1])].reshape(n_cfg, n, 3)

        sl = slice(lo, hi)
        out["combined"][sl] = (SCORE_WEIGHTS["target_weight"] * treward.mean(axis=1)
                               - SCORE_WEIGHTS["wall_weight"] * wall.mean(axis=1)
                               - SCORE_WEIGHTS["lost_weight"] * lost.mean(axis=1))
        out["tdist_mean"][sl] = tdist.mean(axis=1)
        out["tdist_min"][sl] = tdist.min(axis=1)
        out["wall_frac"][sl] = wall.mean(axis=1)
        out["lost_frac"][sl] = lost.mean(axis=1)
        out["hit_frac"][sl] = reached.mean(axis=1)
        out["unresolved_frac"][sl] = 1.0 - (reached | lost | wall).mean(axis=1)
        out["final_x"][sl], out["final_y"][sl], out["final_z"][sl] = final.mean(axis=1).T
    return out


def report_configs(labels, diag, header, top=None):
    log(f"\n  {'config':<24} {'score':>7} {'d_mean':>7} {'d_min':>7} {'hit%':>5} "
        f"{'wall%':>6} {'lost%':>6} {'unres%':>6}  {'mean final (x,y,z)':>22}")
    order = np.argsort(diag["combined"])[::-1] if top else range(len(labels))
    shown = list(order)[:top] if top else order
    for i in shown:
        log(f"  {labels[i]:<24} {diag['combined'][i]:>+7.3f} {diag['tdist_mean'][i]:>7.1f} "
            f"{diag['tdist_min'][i]:>7.1f} {diag['hit_frac'][i]*100:>5.0f} "
            f"{diag['wall_frac'][i]*100:>6.0f} {diag['lost_frac'][i]*100:>6.0f} "
            f"{diag['unresolved_frac'][i]*100:>6.0f}  "
            f"({diag['final_x'][i]:6.1f},{diag['final_y'][i]:6.1f},{diag['final_z'][i]:6.1f})")


def main():
    t_start = time.time()
    print("Loading field map (19 basis CSVs)...")
    bfm = BatchBasisFieldMap.from_directory(HERE, n_electrodes=N_ELECTRODES)
    print("Building wall collision index...")
    wall_index = build_wall_index(HERE, exclude=WALL_EXCLUDE, target_spacing=2.0)
    species = IonSpecies(mass=28 * 1.66053906660e-27, charge=1.602176634e-19)
    start_positions, start_velocities = make_beam(
        N=BEAM_N, species=species, start_point=list(START_POINT),
        mean_energy_eV=15.0, std_energy_eV=0.42466, half_angle_deg=15.0, seed=42,
    )

    log("=" * 78)
    log("BENDER FIELD ANALYSIS -- shared analysis for options #2/#3")
    log("=" * 78)

    centroids = locate_electrodes(bfm, BENDER)
    impulses = probe_per_volt_fields(bfm, BENDER, np.mean([centroids[e] for e in BENDER], axis=0))
    pattern, bender_center = derive_pattern(centroids, impulses)

    # --- Stage 4: coarse strength scan + single-electrode probes ---------
    s_coarse = np.linspace(-1000.0, 1000.0, 41)
    labels, rows = [], []
    for s in s_coarse:
        v = base_voltages(1)[0]
        v += s * pattern
        labels.append(f"pattern s={s:+.0f}")
        rows.append(v)
    for e in BENDER:
        for sign in (+500.0, -500.0):
            v = base_voltages(1)[0]
            v[e - 1] = sign
            labels.append(f"only V{e}={sign:+.0f}")
            rows.append(v)
    voltages_batch = np.vstack([rows])

    log(f"\n--- Stage 4: coarse RK4 scan ({len(labels)} configs, {BEAM_N} particles x "
        f"{NUM_STEPS} steps) ---")
    t0 = time.time()
    diag = fly_batch(bfm, wall_index, voltages_batch, start_positions, start_velocities, species)
    log(f"  flew in {time.time()-t0:.1f}s")
    report_configs(labels, diag, "coarse")

    n_s = len(s_coarse)
    best_i = int(np.argmax(diag["combined"][:n_s]))
    best_s = s_coarse[best_i]
    log(f"\n  best pattern strength (coarse): s={best_s:+.0f}  "
        f"score={diag['combined'][best_i]:+.3f}  mean miss={diag['tdist_mean'][best_i]:.1f}mm")

    # --- Stage 5: refine s, then sweep einzel lenses at best s -----------
    ds = s_coarse[1] - s_coarse[0]
    s_fine = np.linspace(best_s - ds, best_s + ds, 21)
    fine_rows = [base_voltages(1)[0] + s * pattern for s in s_fine]
    fine_labels = [f"pattern s={s:+.1f}" for s in s_fine]

    log(f"\n--- Stage 5a: refine s in [{s_fine[0]:+.0f}, {s_fine[-1]:+.0f}] ---")
    t0 = time.time()
    diag_f = fly_batch(bfm, wall_index, np.vstack(fine_rows), start_positions, start_velocities, species)
    log(f"  flew in {time.time()-t0:.1f}s")
    report_configs(fine_labels, diag_f, "fine", top=8)
    best_fi = int(np.argmax(diag_f["combined"]))
    best_s_fine = s_fine[best_fi]

    einzel_vals = np.array([-600.0, -300.0, 0.0, 300.0, 600.0])
    ez_rows, ez_labels = [], []
    for v3 in einzel_vals:
        for v6 in einzel_vals:
            v = base_voltages(1)[0] + best_s_fine * pattern
            v[3 - 1] = v3
            v[6 - 1] = v6
            ez_rows.append(v)
            ez_labels.append(f"V3={v3:+.0f} V6={v6:+.0f}")

    log(f"\n--- Stage 5b: einzel sweep (V3, V6) at s={best_s_fine:+.1f} ---")
    t0 = time.time()
    diag_e = fly_batch(bfm, wall_index, np.vstack(ez_rows), start_positions, start_velocities, species)
    log(f"  flew in {time.time()-t0:.1f}s")
    report_configs(ez_labels, diag_e, "einzel", top=8)

    best_ei = int(np.argmax(diag_e["combined"]))
    best_v = np.vstack(ez_rows)[best_ei]
    if diag_e["combined"][best_ei] < diag_f["combined"][best_fi]:
        # Einzel sweep didn't beat lenses-at-zero -- keep the simpler point.
        best_v = np.vstack(fine_rows)[best_fi]
        log("  (einzel sweep did not improve on V3=V6=0 -- keeping lenses at 0)")

    starting_point = {f"V{e}": float(best_v[e - 1]) for e in sorted(op.OPTIMIZE)}
    log("\n" + "=" * 78)
    log("DERIVED STARTING POINT (for orchestrator.py enqueue / option #3):")
    for k, v in starting_point.items():
        log(f"   {k} = {v:+.1f}")
    log("=" * 78)

    with open(HERE / "derived_starting_point.json", "w") as f:
        json.dump(starting_point, f, indent=2)
    log("saved: derived_starting_point.json")

    # --- plot -------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(s_coarse, diag["combined"][:n_s], "o-", label="coarse")
    axes[0].plot(s_fine, diag_f["combined"], ".-", label="fine")
    axes[0].set_ylabel("combined_score")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(s_coarse, diag["tdist_mean"][:n_s], "o-", label="mean miss (coarse)")
    axes[1].plot(s_fine, diag_f["tdist_mean"], ".-", label="mean miss (fine)")
    axes[1].plot(s_coarse, diag["tdist_min"][:n_s], "s--", alpha=0.6, label="best-particle miss (coarse)")
    axes[1].set_xlabel("pattern strength s (V);  V9..V12 = s * pattern")
    axes[1].set_ylabel("distance to detector (mm)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle("Quadrupole-bender pattern scan (batched RK4)")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "bender_pattern_scan.png", dpi=140)
    log("saved: outputs/bender_pattern_scan.png")

    with open(OUTPUTS / "bender_analysis_results.txt", "w") as f:
        f.write("\n".join(REPORT_LINES) + "\n")
    log(f"saved: bender_analysis_results.txt  (total {time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
