"""
validar_grilla_fina.py
=======================

LA PRUEBA DE FUEGO del arreglo del cuadrupolo. Cinco configs con verdad
SIMION medida ayer (registro_corridas.jsonl) se vuelan bajo:

  VIEJA fisica: bases 50^3 (~10mm) + paredes STL (residuo 11-15mm)
  NUEVA fisica: CampoDual (fino 2.5mm en el quad) + ParedesPA (metal
                del propio PA, electrodos 1/2/19 excluidos)

Verdades a reproducir (SIMION, 500 iones):
  BASE                  51 al plano   J=0.745
  quad x1.10            0  al plano   (haz muerto -- el RK4 viejo decia "sin cambio")
  V3=250 + quad x1.25   0  al plano   (muerto)
  V3=250 + quad x0.70   368 al plano  J=0.696  (el RK4 viejo decia "desastre")
  RECORD (+V6=-750)     384 al plano  43 hits  J=0.56 (viejo: wall=85%, reach=0)

Si la nueva fisica ordena vivos vs muertos correctamente, el punto ciego
esta arreglado y se promueve a la raiz.

Correr:  python playpen/validar_grilla_fina.py
"""

import pathlib
import sys
import time

import numpy as np
import optuna

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
from caracterizador import cinematica_en_plano
import optimizer as orch
orch.N_ELECTRODES = 19
from validate_rk4_filter import collect_archived_trials
from physics import CampoDual, ParedesPA

N_PART, PASOS, DT = 200, 3000, 1.0e-8
QUAD = (9, 10, 11, 12)

# (nombre, v3, escala_quad, v6, verdad SIMION "al plano" de 500)
CASOS = [
    ("BASE",                None, None,  None,   51),
    ("quad x1.10",          None, 1.10,  None,    0),
    ("V3=250 quad x1.25",  250.0, 1.25,  None,    0),
    ("V3=250 quad x0.70",  250.0, 0.70,  None,  368),
    ("RECORD (V6=-750)",   250.0, 0.70, -750.0, 384),
]


def construir_configs():
    voltages, hits = collect_archived_trials()
    base = voltages[int(np.argmax(hits))].copy()
    out = []
    for nombre, v3, esc, v6, verdad in CASOS:
        v = base.copy()
        if v3 is not None:
            v[2] = v3
        if esc is not None:
            for e in QUAD:
                v[e - 1] = base[e - 1] * esc
        if v6 is not None:
            v[5] = v6
        out.append((nombre, v, verdad))
    return out


def volar(bfm, wall_index, configs, species, sp, sv):
    volts = np.stack([v for _, v, _ in configs])
    n_cfg = volts.shape[0]
    bfm.set_voltages_batch(volts)
    beam, ci = make_batch_beam(species, sp, sv, n_cfg)
    traj = BatchTrajectory(beam, ci)
    BatchRK4Integrator(bfm, ci).integrate(traj, dt=DT, num_steps=PASOS)
    scorer = BeamProgressScorer(
        bfm=bfm, Trajectory=traj, dt=DT, num_steps=PASOS,
        detector_bbox=orch.DETECTOR_BBOX, wall_index=wall_index, wall_hit_margin=1.5,
        wall_check_midpoints=False, wall_check_stride=3)
    r = scorer.combined_score(volts, **orch.SCORE_WEIGHTS)

    pos = r["positions"].reshape(-1, n_cfg, N_PART, 3)
    stop = r["stop_idx"].reshape(n_cfg, N_PART)
    pared = r["hit_wall_step"].reshape(n_cfg, N_PART)
    wall_fr = r["hit_wall"].reshape(n_cfg, N_PART).mean(axis=1)
    reach = r["reached_target"].reshape(n_cfg, N_PART).mean(axis=1)

    filas = []
    for c in range(n_cfg):
        vel_c = np.asarray([s.velocity[c * N_PART:(c + 1) * N_PART] for s in traj.states],
                           dtype=np.float32)
        f = cinematica_en_plano(pos[:, c], vel_c, stop_idx=stop[c], paso_pared=pared[c],
                                z_plano=390.0, eje=2, direccion=+1)
        filas.append(dict(wall=float(wall_fr[c]), reach=float(reach[c]),
                          n390=f["n_cruzan"]))
    return filas


def main():
    configs = construir_configs()
    species = IonSpecies(mass=28 * 1.66053906660e-27, charge=1.602176634e-19)
    sp, sv = make_beam(N=N_PART, species=species, start_point=[395.0, 75.0, 77.0],
                       mean_energy_eV=15.0, std_energy_eV=0.42466, half_angle_deg=15.0, seed=42)

    print("--- fisica VIEJA (bases 50^3 + STL) ---")
    t0 = time.time()
    bfm_v = BatchBasisFieldMap.from_directory(ROOT, n_electrodes=orch.N_ELECTRODES)
    wall_v = build_wall_index(ROOT, exclude=orch.WALL_EXCLUDE, target_spacing=2.0)
    viejas = volar(bfm_v, wall_v, configs, species, sp, sv)
    print(f"  ({time.time() - t0:.0f}s)")
    del bfm_v, wall_v

    print("--- fisica NUEVA (CampoDual + ParedesPA) ---")
    t0 = time.time()
    bfm_n = CampoDual.desde_proyecto(ROOT)
    wall_n = ParedesPA.desde_proyecto(ROOT)
    # sanity: el arranque del haz no debe estar "dentro" de una pared
    d0 = wall_n.distance(np.array([[395.0, 75.0, 77.0], [75.0, 75.0, 77.0]]))
    print(f"  distancia a metal: arranque={d0[0]:.1f}mm, codo del haz={d0[1]:.1f}mm")
    nuevas = volar(bfm_n, wall_n, configs, species, sp, sv)
    print(f"  ({time.time() - t0:.0f}s)")

    print(f"\n===== VIEJA vs NUEVA vs SIMION (n al plano escalado a 500) =====")
    print(f"  {'config':<22} {'wall v/n':>13} {'reach v/n':>11} "
          f"{'n390 v/n (x2.5)':>16} {'SIMION':>7}")
    for (nombre, _, verdad), vie, nue in zip(configs, viejas, nuevas):
        print(f"  {nombre:<22} {vie['wall']:>5.2f}/{nue['wall']:<5.2f} "
              f"{vie['reach']:>5.2f}/{nue['reach']:<5.2f} "
              f"{vie['n390'] * 2.5:>7.0f}/{nue['n390'] * 2.5:<7.0f} {verdad:>7}")
    print("\n  criterio: la NUEVA debe separar vivos (x0.70: ~370-390) de")
    print("  muertos (x1.10, x1.25: 0) y dejar de inventar paredes en el record.")


if __name__ == "__main__":
    main()
