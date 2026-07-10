"""
afinar_foco_postquad.py
========================

EL PASO 2 de la via de colimacion confirmada. arbitrar_colimacion_
simion.py demostro en SIMION real que V3=250 + quad x0.70 transporta
368/500 iones al plano del detector (7x el base) -- pero llegan anchas
(sigma_y 11mm, div 501 mrad) y solo 1-2 caen en la ventana. Ahora hay
368 particulas que ENFOCAR en vez de 51.

Barrido SIMION de V6 (el enfoque del tramo final) sobre ese config,
midiendo hits, cuerpo (n_plano), sigma y offset -- si el centroide esta
corrido, el siguiente knob son las placas V15/V18.

Correr:  python playpen/afinar_foco_postquad.py
"""

import pathlib
import sys

import numpy as np
import optuna

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

optuna.logging.set_verbosity(optuna.logging.WARNING)

from gemelo import GemeloDigital
from validate_rk4_filter import collect_archived_trials

QUAD = (9, 10, 11, 12)
V6_VALS = (-800.0, -600.0, -400.0, -208.0, -100.0, 0.0, 100.0, 250.0, 400.0)


def main():
    voltages, hits = collect_archived_trials()
    base = voltages[int(np.argmax(hits))].copy()

    # el config confirmado: entrada triple + quad suave
    v0 = base.copy()
    v0[2] = 250.0
    for e in QUAD:
        v0[e - 1] = base[e - 1] * 0.70

    tw = GemeloDigital()
    print(f"Config confirmado (V3=250, quad x0.70); barriendo V6 "
          f"(actual {base[5]:.0f})...\n")
    filas = []
    for v6 in V6_VALS:
        v = v0.copy()
        v[5] = v6
        r = tw.evaluar(v)
        f = r.get("features") or {}
        filas.append((v6, r["hits"], f.get("hits", 0), f.get("n_plane", 0),
                      f.get("sigma_x_mm", np.nan), f.get("sigma_y_mm", np.nan),
                      f.get("offset_x_mm", np.nan), f.get("offset_y_mm", np.nan),
                      r["objetivo"]))
        print(f"  V6={v6:>6.0f}: hits={r['hits']:<3} ventana={f.get('hits', 0):<3} "
              f"cuerpo={f.get('n_plane', 0):<4} sigma=({f.get('sigma_x_mm', np.nan):.1f},"
              f"{f.get('sigma_y_mm', np.nan):.1f})mm "
              f"offset=({f.get('offset_x_mm', np.nan):+.1f},{f.get('offset_y_mm', np.nan):+.1f})mm "
              f"J={r['objetivo']:.3f}")

    print("\n===== RESUMEN (ordenado por J_v2) =====")
    print(f"  {'V6':>6} {'hits':>4} {'vent':>4} {'cuerpo':>6} {'sigma_y':>7} "
          f"{'off_x':>6} {'off_y':>6} {'J_v2':>6}")
    for fila in sorted(filas, key=lambda x: x[-1]):
        v6, h, hv, np_, sx, sy, ox, oy, j = fila
        print(f"  {v6:>6.0f} {h:>4} {hv:>4} {np_:>6} {sy:>7.1f} {ox:>+6.1f} {oy:>+6.1f} {j:>6.3f}")
    print("\n  Si el mejor V6 deja el offset lejos de 0, el siguiente barrido es")
    print("  V15/V18 (placas deflectoras) para centrar el haz en la ventana.")


if __name__ == "__main__":
    main()
