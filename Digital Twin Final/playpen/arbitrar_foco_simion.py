"""
arbitrar_foco_simion.py -- PASO 4 del plan de enfoque (2026-07-06)
===================================================================

Los barridos RK4 (fisica dual 1mm) alrededor del trial 26 (J=0.524,
reach 0.92, CERO hits -- el "llega pero no enfoca" en persona)
encontraron candidatos con haz predicho de ~2mm casi centrado:

    quad V11 x0.85            sigma (2.3, 1.6)mm  offset (+1.5, -0.7)
    quad conj V11x0.89/V12x0.97 sigma (2.2, 1.4)  offset (+2.1, -0.7)
    placas V18 -200 (borde!)  mejora puntual, optimo quiza mas alla

SIMION real decide (~6s por vuelo; todo queda en el registro / Task B).
Si alguno supera los 47 hits del record, correr despues
playpen/sembrar_record.py para actualizar punto de partida + estudio.

Correr:  python playpen/arbitrar_foco_simion.py
"""

import pathlib
import sys

import numpy as np
import optuna

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

optuna.logging.set_verbosity(optuna.logging.WARNING)

from gemelo import GemeloDigital

QUAD = (9, 10, 11, 12)


def main():
    tw = GemeloDigital()
    m = tw.mejor()
    base = tw.voltajes_completos(m["voltajes"])
    print(f"Base: trial {m['trial']} (J={m['objetivo']:.3f}, hits={m['hits']})\n")

    def con(quad_f=None, v18=None, v15=None):
        v = base.copy()
        if quad_f:
            for e, f in zip(QUAD, quad_f):
                v[e - 1] = base[e - 1] * f
        if v18 is not None:
            v[17] = v18
        if v15 is not None:
            v[14] = v15
        return v

    candidatos = [
        ("quad V11x0.85",             con(quad_f=(1, 1, 0.85, 1))),
        ("quad conj 0.89/0.97",       con(quad_f=(1.00, 1.01, 0.89, 0.97))),
        ("placas V18=-522",           con(v18=-522.4)),
        ("V11x0.85 + V18=-522",       con(quad_f=(1, 1, 0.85, 1), v18=-522.4)),
        ("V11x0.85 + V18=-622",       con(quad_f=(1, 1, 0.85, 1), v18=-622.4)),
    ]

    resultados = []
    for nombre, v in candidatos:
        print(f"--- {nombre} ---")
        r = tw.evaluar(v)
        f = r.get("features") or {}
        resultados.append((nombre, r))
        print(f"    J={r['objetivo']:.3f}  hits={r['hits']}  "
              f"cuerpo={f.get('n_plane', '?')}  "
              f"sigma=({f.get('sigma_x_mm', float('nan')):.1f},"
              f"{f.get('sigma_y_mm', float('nan')):.1f})mm  "
              f"offset=({f.get('offset_x_mm', float('nan')):+.1f},"
              f"{f.get('offset_y_mm', float('nan')):+.1f})mm\n")

    print("===== VEREDICTO SIMION (record actual: 47 hits) =====")
    for nombre, r in sorted(resultados, key=lambda x: -x[1]["hits"]):
        print(f"  {nombre:<24} hits={r['hits']:<4} J={r['objetivo']:.3f}")


if __name__ == "__main__":
    main()
