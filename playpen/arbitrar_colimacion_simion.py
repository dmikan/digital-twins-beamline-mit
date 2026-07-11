"""
arbitrar_colimacion_simion.py
==============================

EL ARBITRO. El barrido RK4 (scan_v3_quad.py) dejo una tension que solo
SIMION puede resolver: V3=+250 triplica la entrada al cuadrupolo (17->50
limpias) pero el RK4 dice que ninguna escala del quad lo convierte en
llegada... y el MISMO RK4 odia quad x0.70, que historicamente dio 22
hits reales (el record). El RK4 tiene su punto ciego conocido
(sobre-marcado de paredes); estos 4 vuelos reales (~6s c/u) deciden:

  1. BASE      mejor archivado tal cual (control: ruido de refly)
  2. QUAD11    V3 base, quad x1.10   (mejor marginal del RK4, 49.6mm)
  3. ANCHO125  V3=+250, quad x1.25   (entrada triple, mejor variante RK4)
  4. ANCHO070  V3=+250, quad x0.70   (entrada triple + sweet spot
                                      historico de SIMION -- la apuesta
                                      que el RK4 no sabe juzgar)

Cada evaluar() queda ademas registrado con features completas en
studies/registro_corridas.jsonl (dataset Task B).

Correr:  python playpen/arbitrar_colimacion_simion.py
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


def variante(base, v3=None, esc=None):
    v = base.copy()
    if v3 is not None:
        v[2] = v3
    if esc is not None:
        for e in QUAD:
            v[e - 1] = base[e - 1] * esc
    return v


def main():
    voltages, hits = collect_archived_trials()
    base = voltages[int(np.argmax(hits))].copy()
    print(f"Base: mejor archivado ({hits.max():g} hits historicos)\n")

    candidatos = [
        ("BASE (control refly)", variante(base)),
        ("QUAD x1.10 (mejor RK4)", variante(base, esc=1.10)),
        ("V3=250 + quad x1.25", variante(base, v3=250.0, esc=1.25)),
        ("V3=250 + quad x0.70", variante(base, v3=250.0, esc=0.70)),
    ]

    tw = GemeloDigital()
    resultados = []
    for nombre, v in candidatos:
        print(f"--- {nombre} ---")
        r = tw.evaluar(v)
        f = r.get("features") or {}
        resultados.append((nombre, r))
        print(f"    objetivo={r['objetivo']:.3f}  hits={r['hits']}  "
              f"n_plano={f.get('n_plane', '?')}  "
              f"sigma_y={f.get('sigma_y_mm', float('nan')):.2f}mm  "
              f"div_y={f.get('div_y_mrad', float('nan')):.0f}mrad\n")

    print("===== VEREDICTO SIMION =====")
    for nombre, r in sorted(resultados, key=lambda x: -x[1]["hits"]):
        print(f"  {nombre:<26} hits={r['hits']:<4} objetivo={r['objetivo']:.3f}")


if __name__ == "__main__":
    main()
