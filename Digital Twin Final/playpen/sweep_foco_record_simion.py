"""
sweep_foco_record_simion.py -- enfoque, camino SIMION directo (2026-07-06)
==========================================================================

La arbitracion RK4 fallo por dos razones ya corregidas: (1) los barridos
se centraron en trial 26 (0 hits) en vez del record de 47 -- mejor()
elegia por J_v2, que rankeaba mal; (2) el RK4 aun no es fiable en el
enfoque FINAL sobre la cara del detector (predice reach 0.92 donde SIMION
da 0). Asi que el enfoque se afina como se encontro el record: barrido
SIMION directo (~6s/vuelo, verdad de terreno) alrededor del record, en
las DOS perillas nunca tocadas -- placas V15/V18 (punteria) y balance del
cuadrupolo V11 (astigmatismo).

Todo queda en el registro (Task B). Si algo supera 47 hits, correr
playpen/sembrar_record.py para fijar el nuevo punto de partida.

Correr:  python playpen/sweep_foco_record_simion.py
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
    print(f"Record base: trial {m['trial']}, {m['hits']} hits, J={m['objetivo']:.3f}")
    print(f"  V15={base[14]:.0f}, V18={base[17]:.0f}, V11={base[10]:.0f}\n")

    def con(**kw):
        v = base.copy()
        for e, val in kw.items():
            v[int(e[1:]) - 1] = val
        return v

    # tres barridos 1D desde el record (el punto central se vuela una vez)
    barridos = []
    for d in (-200, -100, 100, 200):
        barridos.append((f"V15={base[14] + d:.0f}", con(V15=base[14] + d)))
    for d in (-200, -100, 100, 200):
        barridos.append((f"V18={base[17] + d:.0f}", con(V18=base[17] + d)))
    for fac in (0.70, 0.85, 1.15, 1.30):
        barridos.append((f"V11 x{fac:g} ({base[10] * fac:.0f})", con(V11=base[10] * fac)))
    barridos.insert(0, ("RECORD (control)", base))

    resultados = []
    for nombre, v in barridos:
        r = tw.evaluar(v)
        f = r.get("features") or {}
        resultados.append((nombre, r))
        print(f"  {nombre:<22} hits={r['hits']:<4} J={r['objetivo']:.3f}  "
              f"cuerpo={f.get('n_plane', '?')}  "
              f"sigma=({f.get('sigma_x_mm', float('nan')):.1f},"
              f"{f.get('sigma_y_mm', float('nan')):.1f})  "
              f"off=({f.get('offset_x_mm', float('nan')):+.1f},"
              f"{f.get('offset_y_mm', float('nan')):+.1f})")

    print(f"\n===== TOP por hits (record actual: {m['hits']}) =====")
    for nombre, r in sorted(resultados, key=lambda x: -x[1]["hits"])[:6]:
        print(f"  {nombre:<22} hits={r['hits']:<4} J={r['objetivo']:.3f}")

    mejor = max(resultados, key=lambda x: x[1]["hits"])
    if mejor[1]["hits"] > m["hits"]:
        print(f"\nNUEVO RECORD: {mejor[0]} con {mejor[1]['hits']} hits "
              f"-- correr sembrar_record.py para fijarlo.")
    else:
        print(f"\nEl record se mantiene ({m['hits']} hits). Los barridos 1D no lo "
              f"superaron; el siguiente paso seria una malla 2D V15xV18 o "
              f"entrenar() con el J_v2 corregido.")


if __name__ == "__main__":
    main()
