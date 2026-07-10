"""
comparar_porteros.py
=====================

LA COMPARACION FINAL para la promocion: fisica VIEJA vs NUEVA como
porteros de ranking, EN EL MISMO DATASET Y LA MISMA CORRIDA (la
recoleccion de DBs archivadas fluctua entre corridas -- OneDrive -- asi
que comparar numeros de corridas distintas era invalido; leccion del
falso colapso de la 1mm).

  VIEJA: bases 50^3 + paredes STL, margen 1.5 (la config de produccion)
  NUEVA: CampoDual 1mm (corredor completo) + ParedesPA sin mascaras de
         corredor, margen 0.5 (la config que calibro casi exacta la
         familia record: 388/395 vs 368/384 SIMION)

Ambas puntuan con la formula base del screening (best-10% + transmision
+ pared). Se reporta Spearman vs hits reales y hitters en top-30.

Correr:  python playpen/comparar_porteros.py
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
sys.path.insert(0, str(ROOT / "playpen"))

optuna.logging.set_verbosity(optuna.logging.WARNING)

from physics import IonSpecies
from physics import (
    BatchBasisFieldMap, make_batch_beam, BatchTrajectory, BatchRK4Integrator,
)
from caracterizador import make_beam, BeamProgressScorer
from physics import build_wall_index
import optimizer as orch
orch.N_ELECTRODES = 19
from validate_rk4_filter import collect_archived_trials
from physics import CampoDual, ParedesPA

SCREEN_N, SCREEN_STEPS = 50, 1500


def escanear(bfm, wall, margen, voltages, species, sp, sv, chunk):
    M = len(voltages)
    n_keep = max(1, int(np.ceil(SCREEN_N * orch.SPLAT_TOP_FRACTION)))
    scores = np.empty(M)
    t0 = time.time()
    for lo in range(0, M, chunk):
        hi = min(lo + chunk, M)
        v = voltages[lo:hi]
        bfm.set_voltages_batch(v)
        beam, ci = make_batch_beam(species, sp, sv, v.shape[0])
        traj = BatchTrajectory(beam, ci)
        BatchRK4Integrator(bfm, ci).integrate(traj, dt=orch.DT, num_steps=SCREEN_STEPS)
        scorer = BeamProgressScorer(
            bfm=bfm, Trajectory=traj, dt=orch.DT, num_steps=SCREEN_STEPS,
            detector_bbox=orch.DETECTOR_BBOX, wall_index=wall, wall_hit_margin=margen,
            wall_check_midpoints=False, wall_check_stride=3)
        r = scorer.combined_score(v, **orch.SCORE_WEIGHTS)
        n_cfg = v.shape[0]
        tdist = r["target_distance"].reshape(n_cfg, SCREEN_N)
        reached = r["reached_target"].reshape(n_cfg, SCREEN_N).mean(axis=1)
        wall_fr = r["hit_wall"].reshape(n_cfg, SCREEN_N).mean(axis=1)
        scores[lo:hi] = (np.sort(tdist, axis=1)[:, :n_keep].mean(axis=1)
                         + orch.HIT_TERM_SCALE_MM * (1.0 - reached)
                         + orch.WALL_PENALTY_MM * wall_fr)
        print(f"    configs {lo}-{hi - 1}  ({time.time() - t0:.0f}s)")
    return scores


def main():
    voltages, hits = collect_archived_trials()
    M = len(voltages)
    hitter_idx = set(np.where(hits > 0)[0])
    print(f"Dataset de ESTA corrida: {M} sets, {len(hitter_idx)} hitters "
          f"(fijado para ambos porteros)\n")

    species = IonSpecies(mass=28 * 1.66053906660e-27, charge=1.602176634e-19)
    sp, sv = make_beam(N=SCREEN_N, species=species, start_point=[395.0, 75.0, 77.0],
                       mean_energy_eV=15.0, std_energy_eV=0.42466, half_angle_deg=15.0, seed=42)

    print("--- portero VIEJO (bases 50^3 + STL, margen 1.5) ---")
    bfm = BatchBasisFieldMap.from_directory(ROOT, n_electrodes=orch.N_ELECTRODES)
    wall = build_wall_index(ROOT, exclude=orch.WALL_EXCLUDE, target_spacing=2.0)
    s_viejo = escanear(bfm, wall, 1.5, voltages, species, sp, sv, chunk=50)
    del bfm, wall

    print("--- portero NUEVO (CampoDual 1mm + ParedesPA sin corredores, margen 0.5) ---")
    bfm = CampoDual.desde_proyecto(ROOT)
    wall = ParedesPA.desde_proyecto(ROOT, verbose=False, incluir_corredores=False)
    s_nuevo = escanear(bfm, wall, 0.5, voltages, species, sp, sv, chunk=8)

    print(f"\n===== PORTEROS, MISMO DATASET ({M} sets, {len(hitter_idx)} hitters) =====")
    for nombre, s in (("VIEJO", s_viejo), ("NUEVO", s_nuevo)):
        sr, _ = spearmanr(-s, hits)
        top30 = set(int(i) for i in np.argsort(s)[:30])
        print(f"  {nombre}: Spearman {sr:+.3f}   hitters en top-30: "
              f"{len(top30 & hitter_idx)}/{len(hitter_idx)}")


if __name__ == "__main__":
    main()
