"""
sweep_foco_v2_simion.py -- segundo barrido de enfoque (2026-07-06)
==================================================================

El primer barrido (sweep_foco_record_simion) encontro que las PLACAS son
la palanca: V18=-429 dio 54 hits (nuevo record, +25% sobre 43) y estaba
en el BORDE del barrido, con off_x residual +3.2mm. V11 (balance del
quad) empeoro todo -> el quad ya esta bien.

Este barrido: (a) empuja V18 mas alla del borde, (b) agrega V15 para
centrar el offset_x residual sobre la mejor base V18. SIMION directo.

Correr:  python playpen/sweep_foco_v2_simion.py
"""

import pathlib
import sys

import numpy as np
import optuna

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

optuna.logging.set_verbosity(optuna.logging.WARNING)

from gemelo import GemeloDigital


def main():
    tw = GemeloDigital()
    m = tw.mejor(por="hits")
    base = tw.voltajes_completos(m["voltajes"])
    # base54 = la familia record con la placa ganadora del barrido 1
    base[17] = -429.0
    print(f"Base = record + V18=-429 (54 hits en barrido 1)\n")

    def con(**kw):
        v = base.copy()
        for e, val in kw.items():
            v[int(e[1:]) - 1] = val
        return v

    pruebas = [
        ("V18=-429 (base54)",     base.copy()),
        ("V18=-529",              con(V18=-529)),
        ("V18=-629",              con(V18=-629)),
        ("V18=-729",              con(V18=-729)),
        ("V18=-529, V15=-142",    con(V18=-529, V15=-142)),
        ("V18=-529, V15=-242",    con(V18=-529, V15=-242)),
        ("V18=-429, V15=-142",    con(V15=-142)),
        ("V18=-429, V15=-242",    con(V15=-242)),
        ("V18=-629, V15=-242",    con(V18=-629, V15=-242)),
    ]

    resultados = []
    for nombre, v in pruebas:
        r = tw.evaluar(v)
        f = r.get("features") or {}
        resultados.append((nombre, r, v))
        print(f"  {nombre:<22} hits={r['hits']:<4} J={r['objetivo']:.3f}  "
              f"cuerpo={f.get('n_plane', '?')}  "
              f"sigma=({f.get('sigma_x_mm', float('nan')):.1f},"
              f"{f.get('sigma_y_mm', float('nan')):.1f})  "
              f"off=({f.get('offset_x_mm', float('nan')):+.1f},"
              f"{f.get('offset_y_mm', float('nan')):+.1f})")

    print(f"\n===== TOP por hits =====")
    for nombre, r, _ in sorted(resultados, key=lambda x: -x[1]["hits"])[:5]:
        print(f"  {nombre:<22} hits={r['hits']:<4} J={r['objetivo']:.3f}")

    mejor = max(resultados, key=lambda x: x[1]["hits"])
    v = mejor[2]
    print(f"\nMEJOR: {mejor[0]} -> {mejor[1]['hits']} hits")
    print("  voltajes: {" + ", ".join(
        f"{e}: {v[e-1]:.1f}" for e in (3, 6, 9, 10, 11, 12, 15, 18)) + "}")


if __name__ == "__main__":
    main()
