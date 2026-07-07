"""
validar_ranking_fisica_nueva.py
================================

EL NUMERO QUE DECIDE LA PROMOCION A RAIZ. La fisica nueva (CampoDual de
corredor completo + ParedesPA) ya demostro que ordena bien los 5 casos
canonicos (record arriba, muertos abajo -- la vieja los tenia
invertidos). Ahora la funcion real del portero: rankear los 150 configs
archivados y ver cuantos de los 14 hitters reales quedan en el top-30.

Referencias (fisica vieja, medidas 2026-07-05):
    sin momento           Spearman +0.303   5/14
    con momento z390 W=20 Spearman +0.353   8/14

Correr:  python playpen/validar_ranking_fisica_nueva.py
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
sys.path.insert(0, str(ROOT / "playpen"))

optuna.logging.set_verbosity(optuna.logging.WARNING)

from physics import IonSpecies
from caracterizador import make_beam
import optimizer as orch
orch.N_ELECTRODES = 19
from validate_rk4_filter import collect_archived_trials
from physics import CampoDual, ParedesPA

SCREEN_N, SCREEN_STEPS = 50, 1500
CHUNK = 25   # los mapas finos suben la memoria por config: chunk mas chico


def main():
    voltages, hits = collect_archived_trials()
    M = len(voltages)

    print("Cargando fisica NUEVA (CampoDual + ParedesPA)...")
    bfm = CampoDual.desde_proyecto(ROOT)
    wall = ParedesPA.desde_proyecto(ROOT, verbose=False)
    species = IonSpecies(mass=28 * 1.66053906660e-27, charge=1.602176634e-19)
    sp, sv = make_beam(N=SCREEN_N, species=species, start_point=[395.0, 75.0, 77.0],
                       mean_energy_eV=15.0, std_energy_eV=0.42466, half_angle_deg=15.0, seed=42)

    base = np.empty(M)
    kin = np.empty(M)
    print(f"Re-escaneando {M} sets con la fisica nueva "
          f"({SCREEN_N} particulas x {SCREEN_STEPS} pasos, chunk {CHUNK})...")
    t0 = time.time()
    for lo in range(0, M, CHUNK):
        hi = min(lo + CHUNK, M)
        total, b, k = orch.rk4_score_chunk(
            bfm, wall, voltages[lo:hi], sp, sv, species, orch.DT, SCREEN_STEPS,
            detalle=True)
        base[lo:hi], kin[lo:hi] = b, k
        print(f"  configs {lo}-{hi - 1}  ({time.time() - t0:.0f}s)")

    hitter_idx = set(np.where(hits > 0)[0])
    print(f"\n===== RANKING (fisica NUEVA) vs HITS SIMION (n={M}, {len(hitter_idx)} hitters) =====")
    print(f"  {'variante':<34} {'Pearson':>8} {'Spearman':>9}   hitters en top-30")
    for nombre, s in (("nueva, sin momento", base),
                      ("nueva, con momento W=20", base + orch.KIN_PENALTY_MM * kin)):
        pr, _ = pearsonr(-s, hits)
        sr, _ = spearmanr(-s, hits)
        top30 = set(int(i) for i in np.argsort(s)[:30])
        print(f"  {nombre:<34} {pr:>+8.3f} {sr:>+9.3f}   {len(top30 & hitter_idx)}/{len(hitter_idx)}")
    print("  referencias fisica vieja:            +0.157    +0.303   5/14 (sin momento)")
    print("                                       +0.166    +0.353   8/14 (con momento)")

    # el ranking del config record y su familia (sembrados en el estudio):
    # con la fisica vieja el screening los enterraba (score ~136mm).
    con = base + orch.KIN_PENALTY_MM * kin
    print(f"\n  score del mejor archivado (BASE, 5 hits): "
          f"{con[int(np.argmax(hits))]:.1f}mm (rank {int((con < con[int(np.argmax(hits))]).sum()) + 1}/{M})")


if __name__ == "__main__":
    main()
