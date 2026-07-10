"""
scan_forma_quad.py -- PASO 3 del plan de enfoque (2026-07-06)
==============================================================

ASTIGMATISMO: el plano y llega siempre peor que el x (sigma, emitancia,
aberracion). Hasta hoy el cuadrupolo solo se escalo UNIFORMEMENTE
(x0.70) -- sus 4 rodillos (V9-12) nunca se re-balancearon
individualmente, que es la perilla que iguala los dos planos.

Dos bloques (fisica dual 1mm, 200 particulas por config):
  a) barrido por-rodillo: cada V9..V12 solo, factores 0.85/0.925/1.075/1.15
  b) 30 perturbaciones conjuntas gaussianas (sigma 7% por rodillo, seed 7)

Rankeado por proxy de hits; la columna sig_x/sig_y muestra el balance.
Correr:  python playpen/scan_forma_quad.py
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from foco_comun import QUAD, base_del_estudio, volar_configs, tabla

FACTORES = (0.85, 0.925, 1.075, 1.15)
N_CONJUNTAS = 30
STD_CONJUNTA = 0.07


def main():
    base = base_del_estudio()

    nombres, volts = ["base (record)"], [base.copy()]
    # a) por-rodillo
    for e in QUAD:
        for f in FACTORES:
            v = base.copy()
            v[e - 1] = base[e - 1] * f
            nombres.append(f"V{e} x{f:g}")
            volts.append(v)
    # b) conjuntas
    rng = np.random.default_rng(7)
    for j in range(N_CONJUNTAS):
        v = base.copy()
        fs = 1.0 + rng.normal(0.0, STD_CONJUNTA, len(QUAD))
        for e, f in zip(QUAD, fs):
            v[e - 1] = base[e - 1] * f
        nombres.append("conj " + "/".join(f"{f:.2f}" for f in fs))
        volts.append(v)
    volts = np.array(volts)
    print(f"Barrido de forma del quad: {len(volts)} configs "
          f"({len(QUAD) * len(FACTORES)} por-rodillo + {N_CONJUNTAS} conjuntas + base)")

    met = volar_configs(volts)
    orden = tabla(nombres, met, top=15)

    i = orden[0]
    if i == 0:
        print("\nEl record sigue siendo el mejor: el quad ya esta bien balanceado "
              "(el enfoque restante es de placas/V6, no de forma).")
    else:
        v = volts[i]
        print(f"\nMEJOR: {nombres[i]} -- voltajes quad: "
              + ", ".join(f"V{e}={v[e - 1]:.0f}" for e in QUAD)
              + " -- candidato para arbitraje SIMION")


if __name__ == "__main__":
    main()
