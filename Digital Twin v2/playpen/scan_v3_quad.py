"""
scan_v3_quad.py
================

FASE 3 del ataque a la colimacion pre-cuadrupolo. Lo aprendido:
  - scan_colimacion_prequad.py: V3=+250 TRIPLICA las particulas limpias
    en la entrada del quad (17 -> 50) pero el haz llega ANCHO (sigma 16mm
    vs 6mm) y el cuadrupolo actual, afinado para el haz angosto, lo
    pierde (score linea completa 69-217 vs 50 del base).
  - V6 es plano rio arriba: no es perilla pre-quad.
  - Historico: quad x0.70 dio 22 hits (record de todo el proyecto).

Hipotesis: V3 y el cuadrupolo estan ACOPLADOS -- la entrada triple
necesita el quad re-escalado. Barrido: V3 x factor de escala de V9-12
(sobre el mejor config), linea completa, buscando conservar el caudal
de entrada Y doblarlo bien.

Correr:  python playpen/scan_v3_quad.py
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

from RK4_sim_basis import IonSpecies
from RK4_sim_basis_batch import BatchBasisFieldMap
from beam_progress_score import make_beam
from electrode_geometry import build_wall_index
from caracterizador import cinematica_en_plano, puntaje_cinematico
import orchestrator as orch
from validate_rk4_filter import collect_archived_trials
from scan_colimacion_prequad import volar, medir_prequad, N_PART, PASOS_LARGOS

V3_VALS = (None, 0.0, 250.0, 500.0)          # None = V3 del config base
QUAD_SCALES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4)
QUAD_ELECTRODES = (9, 10, 11, 12)


def main():
    voltages, hits = collect_archived_trials()
    i_best = int(np.argmax(hits))
    base_v = voltages[i_best].copy()
    print(f"Config base: mejor archivado ({hits[i_best]:g} hits) -- V3={base_v[2]:.0f}, "
          f"quad=({', '.join(f'{base_v[e-1]:.0f}' for e in QUAD_ELECTRODES)})")

    print("Cargando campo + paredes...")
    bfm = BatchBasisFieldMap.from_directory(ROOT, n_electrodes=orch.N_ELECTRODES)
    wall_index = build_wall_index(ROOT, exclude=orch.WALL_EXCLUDE, target_spacing=2.0)
    species = IonSpecies(mass=28 * 1.66053906660e-27, charge=1.602176634e-19)
    sp, sv = make_beam(N=N_PART, species=species, start_point=[395.0, 75.0, 77.0],
                       mean_energy_eV=15.0, std_energy_eV=0.42466, half_angle_deg=15.0, seed=42)

    combos, volts = [], []
    for v3 in V3_VALS:
        for esc in QUAD_SCALES:
            v = base_v.copy()
            if v3 is not None:
                v[2] = v3
            for e in QUAD_ELECTRODES:
                v[e - 1] = base_v[e - 1] * esc
            combos.append((base_v[2] if v3 is None else v3, esc))
            volts.append(v)
    volts = np.array(volts)
    print(f"Barrido: {len(volts)} combos (V3 x escala quad), linea completa "
          f"({N_PART} particulas x {PASOS_LARGOS} pasos)...")

    n_keep = max(1, int(np.ceil(N_PART * orch.SPLAT_TOP_FRACTION)))

    def medidor(pos, vel_c, stop, pared, extra):
        r, c, n_cfg = extra
        tdist = r["target_distance"].reshape(n_cfg, N_PART)[c]
        reached = r["reached_target"].reshape(n_cfg, N_PART)[c].mean()
        wall = r["hit_wall"].reshape(n_cfg, N_PART)[c].mean()
        score = (np.sort(tdist)[:n_keep].mean()
                 + orch.HIT_TERM_SCALE_MM * (1.0 - reached)
                 + orch.WALL_PENALTY_MM * wall)
        f390 = cinematica_en_plano(pos, vel_c, stop_idx=stop, paso_pared=pared,
                                   z_plano=390.0, eje=2, direccion=+1)
        m = medir_prequad(pos, vel_c, stop, pared)
        return dict(score_mm=float(score), reached=float(reached),
                    n_390=f390["n_cruzan"], kin_390=puntaje_cinematico(f390),
                    sigma_390=f390["sigma_x_mm"], **m)

    t0 = time.time()
    res = volar(bfm, wall_index, species, sp, sv, volts, PASOS_LARGOS, medidor)
    print(f"  barrido en {time.time() - t0:.0f}s")

    orden = np.argsort([m["score_mm"] for m in res])
    print(f"\n===== V3 x ESCALA QUAD, ordenado por score RK4 (menor = mejor) =====")
    print(f"  {'V3':>7} {'quad':>5}  {'n_quad':>6} {'n_z390':>6} {'kin390':>7} "
          f"{'reached':>7} {'score (mm)':>10}")
    for i in orden:
        v3, esc = combos[i]
        m = res[i]
        es_base = abs(esc - 1.0) < 1e-9 and abs(v3 - base_v[2]) < 1e-9
        print(f"  {v3:>7.0f} {esc:>5.2f}  {m['n_quad']:>6.0f} {m['n_390']:>6.0f} "
              f"{m['kin_390']:>7.3f} {m['reached']:>7.3f} {m['score_mm']:>10.1f}"
              + ("   <- BASE actual" if es_base else ""))

    mejor = orden[0]
    v3, esc = combos[mejor]
    print(f"\n  MEJOR: V3={v3:.0f}, quad x{esc:g} -- voltajes completos para gemelo.evaluar():")
    v = volts[mejor]
    print("  {" + ", ".join(f"{e}: {v[e-1]:.1f}" for e in sorted(
        list(QUAD_ELECTRODES) + [3, 6, 15, 18])) + "}")


if __name__ == "__main__":
    main()
