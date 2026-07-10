"""
calibrar_planos_kin.py
=======================

El promedio plano de los 5 planos de KIN_PLANES diluyo la senal (5/14
hitters en top-30 vs 8/14 con el plano unico z=390). Antes de elegir
pesos a ojo: medir QUE APORTA CADA PLANO por separado contra los hits
SIMION archivados, y probar combinaciones ponderadas.

Un vuelo RK4 por chunk (mismo que el screening); por config se calcula
el puntaje cinematico en cada plano por separado. Luego cada plano y
cada combo se evalua como base + W*kin contra los hits reales.

Correr:  python playpen/calibrar_planos_kin.py
"""

import pathlib
import sys
import time

import numpy as np
import optuna
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

optuna.logging.set_verbosity(optuna.logging.WARNING)

from RK4_sim_basis import IonSpecies
from RK4_sim_basis_batch import (
    BatchBasisFieldMap, make_batch_beam, BatchTrajectory, BatchRK4Integrator,
)
from beam_progress_score import make_beam, BeamProgressScorer
from electrode_geometry import build_wall_index
from caracterizador import cinematica_en_plano, puntaje_cinematico
import orchestrator as orch
from validate_rk4_filter import collect_archived_trials

SCREEN_N, SCREEN_STEPS, CHUNK, W = 50, 1500, 50, 20.0

PLANOS = orch.KIN_PLANES   # (eje, direccion, coordenada)
ETIQ = [f"{'x' if e == 0 else 'z'}={c:g}" for e, d, c in PLANOS]


def main():
    voltages, hits = collect_archived_trials()
    print("Cargando campo + paredes...")
    bfm = BatchBasisFieldMap.from_directory(ROOT, n_electrodes=orch.N_ELECTRODES)
    wall_index = build_wall_index(ROOT, exclude=orch.WALL_EXCLUDE, target_spacing=2.0)
    species = IonSpecies(mass=28 * 1.66053906660e-27, charge=1.602176634e-19)
    sp, sv = make_beam(N=SCREEN_N, species=species, start_point=[395.0, 75.0, 77.0],
                       mean_energy_eV=15.0, std_energy_eV=0.42466, half_angle_deg=15.0, seed=42)

    M = len(voltages)
    n_pl = len(PLANOS)
    base = np.empty(M)
    kin_pl = np.empty((n_pl, M))
    n_keep = max(1, int(np.ceil(SCREEN_N * orch.SPLAT_TOP_FRACTION)))

    print(f"Volando {M} sets, midiendo {n_pl} planos por config...")
    t0 = time.time()
    for lo in range(0, M, CHUNK):
        hi = min(lo + CHUNK, M)
        chunk = voltages[lo:hi]
        n_cfg = chunk.shape[0]

        bfm.set_voltages_batch(chunk)
        beam, ci = make_batch_beam(species, sp, sv, n_cfg)
        traj = BatchTrajectory(beam, ci)
        BatchRK4Integrator(bfm, ci).integrate(traj, dt=orch.DT, num_steps=SCREEN_STEPS)
        scorer = BeamProgressScorer(
            bfm=bfm, Trajectory=traj, dt=orch.DT, num_steps=SCREEN_STEPS,
            detector_bbox=orch.DETECTOR_BBOX, wall_index=wall_index, wall_hit_margin=1.5,
            wall_check_midpoints=False, wall_check_stride=3)
        r = scorer.combined_score(chunk, **orch.SCORE_WEIGHTS)

        tdist = r["target_distance"].reshape(n_cfg, SCREEN_N)
        reached = r["reached_target"].reshape(n_cfg, SCREEN_N).mean(axis=1)
        wall = r["hit_wall"].reshape(n_cfg, SCREEN_N).mean(axis=1)
        base[lo:hi] = (np.sort(tdist, axis=1)[:, :n_keep].mean(axis=1)
                       + orch.HIT_TERM_SCALE_MM * (1.0 - reached)
                       + orch.WALL_PENALTY_MM * wall)

        pos = r["positions"].reshape(-1, n_cfg, SCREEN_N, 3)
        stop_r = r["stop_idx"].reshape(n_cfg, SCREEN_N)
        pared_r = r["hit_wall_step"].reshape(n_cfg, SCREEN_N)
        for c in range(n_cfg):
            sl = slice(c * SCREEN_N, (c + 1) * SCREEN_N)
            vel_c = np.asarray([s.velocity[sl] for s in traj.states], dtype=np.float32)
            for j, (eje, d, coord) in enumerate(PLANOS):
                f = cinematica_en_plano(pos[:, c], vel_c, stop_idx=stop_r[c],
                                        paso_pared=pared_r[c], z_plano=coord,
                                        eje=eje, direccion=d)
                kin_pl[j, lo + c] = puntaje_cinematico(f)
        print(f"  configs {lo}-{hi - 1}  ({time.time() - t0:.0f}s)")

    hitter_idx = set(np.where(hits > 0)[0])
    hit_mask = hits > 0

    def evalua(nombre, kin):
        s = base + W * kin
        sr, _ = spearmanr(-s, hits)
        top30 = set(int(i) for i in np.argsort(s)[:30])
        sep = f"{kin[hit_mask].mean():.3f}/{kin.mean():.3f}"
        print(f"  {nombre:<38} {sr:>+8.3f}   {len(top30 & hitter_idx):>2}/{len(hitter_idx)}"
              f"     {sep}")

    print(f"\n===== APORTE POR PLANO (base + {W:g}*kin, n={M}, {len(hitter_idx)} hitters) =====")
    print(f"  {'variante':<38} {'Spearman':>8}   top-30   kin hitters/todos")
    evalua("sin momento (solo base)", np.zeros(M))
    for j in range(n_pl):
        evalua(f"solo plano {ETIQ[j]}", kin_pl[j])

    print("  --- combinaciones ---")
    evalua("promedio 5 planos (lo que fallo)", kin_pl.mean(axis=0))
    evalua("promedio 3 planos z", kin_pl[2:].mean(axis=0))
    evalua("promedio z=300 + z=390", kin_pl[3:].mean(axis=0))
    pesos_tarde = np.array([0.05, 0.05, 0.10, 0.20, 0.60])
    evalua("ponderado tardio (5/5/10/20/60%)", pesos_tarde @ kin_pl)
    pesos_med = np.array([0.10, 0.10, 0.20, 0.25, 0.35])
    evalua("ponderado medio (10/10/20/25/35%)", pesos_med @ kin_pl)

    print("\n  correlacion kin-entre-planos (Spearman):")
    for j in range(n_pl):
        fila = "  ".join(f"{spearmanr(kin_pl[j], kin_pl[k])[0]:+.2f}" for k in range(n_pl))
        print(f"    {ETIQ[j]:<8} {fila}")


if __name__ == "__main__":
    main()
