"""
comparar_ranking_momento.py
============================

Comparacion RANKING SIN MOMENTO (modelo actual) vs CON MOMENTO (propuesto)
contra la verdad de terreno ya pagada: los voltajes archivados con hits
SIMION conocidos (mismo arnes que tools/validate_rk4_filter.py).

Un solo vuelo RK4 por chunk -- orchestrator.rk4_score_chunk(detalle=True)
devuelve (total, base, puntaje_kin), asi que ambos rankings salen de la
MISMA trayectoria y la unica diferencia es el termino cinematico:

    sin momento:  base = best-10% dist + 20*(1-reached) + 50*wall_frac
    con momento:  base + W * puntaje_cinematico(plano z=390, cruces limpios)

Reporta, por variante: Pearson/Spearman contra hits reales y cuantos
hitters reales quedan en el top-30 del ranking (la metrica que decide si
el filtro promueve a SIMION los configs correctos). Ademas el
desplazamiento de rangos entre ambos modelos y el diagnostico del
puntaje cinematico (cuantos configs son medibles en el plano).

Correr:  python playpen/comparar_ranking_momento.py
"""

import pathlib
import sys
import time

import numpy as np
import optuna
from scipy.stats import pearsonr, spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

optuna.logging.set_verbosity(optuna.logging.WARNING)

from RK4_sim_basis import IonSpecies
from RK4_sim_basis_batch import BatchBasisFieldMap
from beam_progress_score import make_beam
from electrode_geometry import build_wall_index
import orchestrator as orch
from validate_rk4_filter import collect_archived_trials

SCREEN_N = 50        # fidelidad etapa A: el filtro que hace el rechazo masivo
SCREEN_STEPS = 1500
CHUNK = 50
PESOS_KIN = (10.0, 20.0, 50.0)   # barrido de W (KIN_PENALTY_MM actual = 20)


def main():
    voltages, hits = collect_archived_trials()
    if len(voltages) < 10:
        raise SystemExit("No hay suficientes trials archivados para validar.")

    print("Cargando mapa de campo...")
    bfm = BatchBasisFieldMap.from_directory(ROOT, n_electrodes=orch.N_ELECTRODES)
    print("Construyendo indice de paredes...")
    wall_index = build_wall_index(ROOT, exclude=orch.WALL_EXCLUDE, target_spacing=2.0)
    species = IonSpecies(mass=28 * 1.66053906660e-27, charge=1.602176634e-19)
    start_positions, start_velocities = make_beam(
        N=SCREEN_N, species=species, start_point=[395.0, 75.0, 77.0],
        mean_energy_eV=15.0, std_energy_eV=0.42466, half_angle_deg=15.0, seed=42)

    M = len(voltages)
    base = np.empty(M)
    kin = np.empty(M)
    print(f"Re-escaneando {M} sets de voltajes ({SCREEN_N} particulas x {SCREEN_STEPS} pasos, "
          f"un vuelo por chunk, dos puntuaciones)...")
    t0 = time.time()
    for lo in range(0, M, CHUNK):
        hi = min(lo + CHUNK, M)
        total, b, k = orch.rk4_score_chunk(
            bfm, wall_index, voltages[lo:hi], start_positions, start_velocities,
            species, orch.DT, SCREEN_STEPS, detalle=True)
        # sanidad del camino integrado por chunks: total == base + W*kin
        assert np.allclose(total, b + orch.KIN_PENALTY_MM * k), "desacople base/kin"
        base[lo:hi], kin[lo:hi] = b, k
        print(f"  configs {lo}-{hi - 1}  ({time.time() - t0:.0f}s)")

    n_planos = len(getattr(orch, "KIN_PLANES", ())) or 1
    variantes = {"sin momento (actual)": base}
    for w in PESOS_KIN:
        etiqueta = (f"con momento {n_planos} planos W={w:g}"
                    + ("  <- KIN_PENALTY_MM" if w == orch.KIN_PENALTY_MM else ""))
        variantes[etiqueta] = base + w * kin

    hitter_idx = set(np.where(hits > 0)[0])
    print(f"\n===== RANKING vs HITS SIMION REALES (n={M} sets, {len(hitter_idx)} hitters) =====")
    print(f"  {'variante':<34} {'Pearson':>8} {'Spearman':>9}   hitters en top-30")
    for nombre, s in variantes.items():
        # score es minimizar -> se niega para correlacionar con hits (mas = mejor)
        pr, _ = pearsonr(-s, hits)
        sr, _ = spearmanr(-s, hits)
        top30 = set(int(i) for i in np.argsort(s)[:30])
        print(f"  {nombre:<34} {pr:>+8.3f} {sr:>+9.3f}   {len(top30 & hitter_idx)}/{len(hitter_idx)}")

    # ---- desplazamiento de rangos entre ambos modelos (W actual) ----
    con = base + orch.KIN_PENALTY_MM * kin
    r_sin = np.argsort(np.argsort(base))
    r_con = np.argsort(np.argsort(con))
    shift = r_con - r_sin
    sr_entre, _ = spearmanr(base, con)
    print(f"\n===== DESPLAZAMIENTO DE RANGOS sin->con momento (W={orch.KIN_PENALTY_MM:g}) =====")
    print(f"  Spearman entre ambos rankings : {sr_entre:+.3f}")
    print(f"  |desplazamiento| medio        : {np.abs(shift).mean():.1f} puestos (max {np.abs(shift).max()})")
    print(f"  configs medibles (algun plano): {(kin < 1.0).sum()}/{M} "
          f"(kin=1.0 = <5 cruces limpios en TODOS los planos)")
    print(f"  puntaje kin de los hitters    : "
          f"{np.mean([kin[i] for i in hitter_idx]):.3f} (medio) vs {kin.mean():.3f} (todos)")

    subieron = np.argsort(shift)[:5]
    print("  los 5 que MAS SUBEN con momento (puesto sin -> con | kin | hits reales):")
    for i in subieron:
        print(f"    #{r_sin[i]:>3} -> #{r_con[i]:>3}  kin={kin[i]:.3f}  hits={hits[i]:g}")
    bajaron = np.argsort(-shift)[:5]
    print("  los 5 que MAS BAJAN:")
    for i in bajaron:
        print(f"    #{r_sin[i]:>3} -> #{r_con[i]:>3}  kin={kin[i]:.3f}  hits={hits[i]:g}")


if __name__ == "__main__":
    main()
