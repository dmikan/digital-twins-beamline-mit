"""
barrer_margen_pared.py
=======================

El margen de contacto (wall_hit_margin=1.5mm) era una compensacion para
los STL corridos 11-15mm. Con ParedesPA (metal del propio PA) el
criterio fisico correcto es "entro al metal": en SIMION un ion que pasa
a 1mm de la superficie sobrevive. Barre margenes {1.5, 1.0, 0.5} con la
fisica nueva y mide las DOS cosas que el portero tiene que hacer bien:

  a) los 5 casos canonicos: record/x0.70 vivos, x1.10/x1.25 muertos,
     BASE ~10% (el sobre-castigo actual de BASE es el sintoma)
  b) ranking de los archivados: Spearman + hitters en top-30.
     OJO con la referencia segun el dataset que collect_archived_trials
     encuentre: sobre 150 sets/14 hitters la vieja fisica dio +0.353 y
     8/14; sobre los 288 sets/61 hitters historicos completos dio +0.458
     y 17/61 (calibracion original de WALL_PENALTY_MM, 2026-07-03).

Un solo vuelo por chunk; el margen solo afecta el scoring, asi que cada
trayectoria se puntua 3 veces.

Correr:  python playpen/barrer_margen_pared.py
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
from physics import make_batch_beam, BatchTrajectory, BatchRK4Integrator
from caracterizador import make_beam, BeamProgressScorer
from caracterizador import cinematica_en_plano
import optimizer as orch
orch.N_ELECTRODES = 19
from validate_rk4_filter import collect_archived_trials
from physics import CampoDual, ParedesPA

MARGENES = (1.5, 1.0, 0.5)
QUAD = (9, 10, 11, 12)
CHUNK = 8   # a 1mm los mapas finos pesan ~26MB/config en f32: chunk chico


def puntuar(traj, bfm, wall, volts, n_part, pasos, margen):
    """Un scorer sobre una trayectoria YA integrada."""
    scorer = BeamProgressScorer(
        bfm=bfm, Trajectory=traj, dt=orch.DT, num_steps=pasos,
        detector_bbox=orch.DETECTOR_BBOX, wall_index=wall, wall_hit_margin=margen,
        wall_check_midpoints=False, wall_check_stride=3)
    r = scorer.combined_score(volts, **orch.SCORE_WEIGHTS)
    n_cfg = volts.shape[0]
    tdist = r["target_distance"].reshape(n_cfg, n_part)
    reached = r["reached_target"].reshape(n_cfg, n_part).mean(axis=1)
    wall_fr = r["hit_wall"].reshape(n_cfg, n_part).mean(axis=1)
    n_keep = max(1, int(np.ceil(n_part * orch.SPLAT_TOP_FRACTION)))
    score = (np.sort(tdist, axis=1)[:, :n_keep].mean(axis=1)
             + orch.HIT_TERM_SCALE_MM * (1.0 - reached)
             + orch.WALL_PENALTY_MM * wall_fr)
    return r, score, wall_fr


def main():
    voltages, hits = collect_archived_trials()
    base_v = voltages[int(np.argmax(hits))].copy()

    print("Cargando fisica nueva...")
    bfm = CampoDual.desde_proyecto(ROOT)
    wall = ParedesPA.desde_proyecto(ROOT, verbose=False)
    species = IonSpecies(mass=28 * 1.66053906660e-27, charge=1.602176634e-19)

    # ---------------- a) casos canonicos (200 particulas) ----------------
    def variante(v3=None, esc=None, v6=None):
        v = base_v.copy()
        if v3 is not None:
            v[2] = v3
        if esc is not None:
            for e in QUAD:
                v[e - 1] = base_v[e - 1] * esc
        if v6 is not None:
            v[5] = v6
        return v

    casos = [("BASE", variante(), 51), ("x1.10", variante(esc=1.10), 0),
             ("250/x1.25", variante(250.0, 1.25), 0),
             ("250/x0.70", variante(250.0, 0.70), 368),
             ("RECORD", variante(250.0, 0.70, -750.0), 384)]
    volts5 = np.stack([v for _, v, _ in casos])
    sp, sv = make_beam(N=200, species=species, start_point=[395.0, 75.0, 77.0],
                       mean_energy_eV=15.0, std_energy_eV=0.42466, half_angle_deg=15.0, seed=42)
    bfm.set_voltages_batch(volts5)
    beam, ci = make_batch_beam(species, sp, sv, 5)
    traj = BatchTrajectory(beam, ci)
    BatchRK4Integrator(bfm, ci).integrate(traj, dt=orch.DT, num_steps=3000)

    print(f"\n===== a) n limpias a z390 (x2.5 -> escala 500) por margen =====")
    print(f"  {'config':<11}" + "".join(f" m={m:<6}" for m in MARGENES) + "  SIMION")
    filas = {n: [] for n, _, _ in casos}
    for m in MARGENES:
        r, _, _ = puntuar(traj, bfm, wall, volts5, 200, 3000, m)
        pos = r["positions"].reshape(-1, 5, 200, 3)
        stop = r["stop_idx"].reshape(5, 200)
        pared = r["hit_wall_step"].reshape(5, 200)
        for c, (nombre, _, _) in enumerate(casos):
            vel_c = np.asarray([s.velocity[c * 200:(c + 1) * 200] for s in traj.states],
                               dtype=np.float32)
            f = cinematica_en_plano(pos[:, c], vel_c, stop_idx=stop[c],
                                    paso_pared=pared[c], z_plano=390.0)
            filas[nombre].append(f["n_cruzan"] * 2.5)
    for (nombre, _, verdad) in casos:
        print(f"  {nombre:<11}" + "".join(f" {v:>7.0f}" for v in filas[nombre])
              + f"  {verdad:>6}")

    # ---------------- b) ranking de los 150 archivados --------------------
    sp, sv = make_beam(N=50, species=species, start_point=[395.0, 75.0, 77.0],
                       mean_energy_eV=15.0, std_energy_eV=0.42466, half_angle_deg=15.0, seed=42)
    M = len(voltages)
    scores = {m: np.empty(M) for m in MARGENES}
    print(f"\nRe-escaneando {M} sets (un vuelo, {len(MARGENES)} margenes)...")
    t0 = time.time()
    for lo in range(0, M, CHUNK):
        hi = min(lo + CHUNK, M)
        chunk = voltages[lo:hi]
        bfm.set_voltages_batch(chunk)
        beam, ci = make_batch_beam(species, sp, sv, chunk.shape[0])
        traj = BatchTrajectory(beam, ci)
        BatchRK4Integrator(bfm, ci).integrate(traj, dt=orch.DT, num_steps=1500)
        for m in MARGENES:
            _, s, _ = puntuar(traj, bfm, wall, chunk, 50, 1500, m)
            scores[m][lo:hi] = s
        print(f"  configs {lo}-{hi - 1}  ({time.time() - t0:.0f}s)")

    hitter_idx = set(np.where(hits > 0)[0])
    print(f"\n===== b) ranking archivados por margen (referencia vieja: +0.353, 8/14) =====")
    for m in MARGENES:
        s = scores[m]
        sr, _ = spearmanr(-s, hits)
        top30 = set(int(i) for i in np.argsort(s)[:30])
        print(f"  margen {m:>4}: Spearman {sr:+.3f}   hitters top-30: "
              f"{len(top30 & hitter_idx)}/{len(hitter_idx)}")


if __name__ == "__main__":
    main()
