"""
bender_ratio_estimate.py
========================

Estimado analitico GRATIS (sin RK4) de la razon k dentro de cada par
diagonal del bender. Reutiliza la maquinaria de bender_field_analysis.py:
carga los 19 campos base, localiza los electrodos 9-12, y para cada uno
integra el impulso en z POR VOLTIO a lo largo del camino nominal.

La idea del par proporcional (no igual): en vez de forzar V_a = V_b dentro
de un par diagonal, permitimos V_b = k * V_a, con k = razon entre cuanto
empuja cada electrodo hacia +z (el detector) por voltio. Ese impulso por
voltio ya lo mide probe_per_volt_fields; aca lo tomamos sobre AMBOS legs
(leg 1 fuente->bender y leg 2 bender->detector), sumando la componente z:

    imp_z(e) = i1[2] + i2[2]        (V*mm/mm por voltio, sobre ambos legs)
    k_par    = imp_z(miembro_2) / imp_z(miembro_1)

Es un estimado LINEAL (asume trayectoria recta por leg, no captura la
curvatura real dentro del bender). Sirve de punto de partida barato y de
chequeo cruzado contra el scan RK4 no lineal.

Run:
    python tools/bender_ratio_estimate.py
"""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from physics import BatchBasisFieldMap
from bender_field_analysis import (
    BENDER, N_ELECTRODES,
    locate_electrodes, probe_per_volt_fields, derive_pattern,
)


def diagonal_pairs(centroids):
    """Reconstruye los dos pares diagonales igual que derive_pattern."""
    center = np.mean([centroids[e] for e in BENDER], axis=0)
    offsets = {e: centroids[e] - center for e in BENDER}
    e0 = BENDER[0]
    others = [e for e in BENDER if e != e0]
    dots = {e: float(np.dot(offsets[e0], offsets[e])) for e in others}
    partner = min(dots, key=dots.get)
    pair_a = (e0, partner)
    pair_b = tuple(e for e in others if e != partner)
    return pair_a, pair_b


def main():
    print("Cargando campo base (19 CSVs)...")
    bfm = BatchBasisFieldMap.from_directory(ROOT, n_electrodes=N_ELECTRODES)

    centroids = locate_electrodes(bfm, BENDER)
    bender_center = np.mean([centroids[e] for e in BENDER], axis=0)
    impulses = probe_per_volt_fields(bfm, BENDER, bender_center)
    pair_a, pair_b = diagonal_pairs(centroids)

    print("\n" + "=" * 70)
    print("ESTIMADO ANALITICO DE k (impulso z por voltio, leg1 + leg2)")
    print("=" * 70)

    # impulso z por voltio sobre ambos legs, por electrodo
    imp_z = {e: impulses[e][0][2] + impulses[e][1][2] for e in BENDER}
    imp_z1 = {e: impulses[e][0][2] for e in BENDER}
    imp_z2 = {e: impulses[e][1][2] for e in BENDER}

    print(f"\n  {'elec':>4} {'leg1 Iz':>12} {'leg2 Iz':>12} {'leg1+2 Iz':>12}")
    for e in BENDER:
        print(f"  {e:>4} {imp_z1[e]:>12.4f} {imp_z2[e]:>12.4f} {imp_z[e]:>12.4f}")

    for name, pair in (("A", pair_a), ("B", pair_b)):
        e1, e2 = pair
        # k relativo al miembro de mayor |impulso| (numerador estable != 0)
        base, other = (e1, e2) if abs(imp_z[e1]) >= abs(imp_z[e2]) else (e2, e1)
        k = imp_z[other] / imp_z[base]
        print(f"\n  Par {name} {pair}:")
        print(f"    base = V{base} (mayor |Iz|),  V{other} = k * V{base}")
        print(f"    k (leg1+leg2) = {k:+.4f}")
        print(f"    k (solo leg1) = {imp_z1[other]/imp_z1[base]:+.4f}   "
              f"k (solo leg2) = {imp_z2[other]/imp_z2[base]:+.4f}")

    print("\n" + "=" * 70)
    print("Nota: k~-1 => par antisimetrico simple (lo que asumia el modelo")
    print("rigido). k lejos de -1 => el par proporcional aporta informacion.")
    print("=" * 70)


if __name__ == "__main__":
    main()
