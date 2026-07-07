"""
scan_placas.py -- PASO 2 del plan de enfoque (2026-07-06)
==========================================================

PUNTERIA: el haz del record llega con centroide corrido (~+3mm en x)
respecto a la ventana de 12x13mm. Las placas deflectoras V15/V18 son la
perilla de apuntado y NUNCA se han barrido (siguen en los valores
heredados de las corridas viejas de Optuna).

Barrido 7x7 alrededor del record (fisica dual 1mm, 200 particulas por
config), rankeado por el proxy de hits (fraccion que entra a la caja) y
mostrando centroide/sigma en el plano pre-detector.

Salida: outputs/scan_placas.png + tabla top-12.
Correr:  python playpen/scan_placas.py
"""

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from foco_comun import ROOT, base_del_estudio, volar_configs, tabla

DELTAS = (-200.0, -100.0, -50.0, 0.0, 50.0, 100.0, 200.0)


def main():
    base = base_del_estudio()
    v15_0, v18_0 = base[14], base[17]

    combos, volts = [], []
    for d15 in DELTAS:
        for d18 in DELTAS:
            v = base.copy()
            v[14] = v15_0 + d15
            v[17] = v18_0 + d18
            combos.append((v15_0 + d15, v18_0 + d18))
            volts.append(v)
    volts = np.array(volts)
    print(f"Barrido V15 x V18: {len(volts)} combos alrededor de "
          f"({v15_0:.0f}, {v18_0:.0f})")

    met = volar_configs(volts)
    nombres = [f"V15={a:.0f}, V18={b:.0f}" for a, b in combos]
    orden = tabla(nombres, met)

    n = len(DELTAS)
    reach = np.array([m["reach"] for m in met]).reshape(n, n)
    offr = np.array([np.hypot(m["off_x"], m["off_y"]) for m in met]).reshape(n, n)
    ejes15 = [v15_0 + d for d in DELTAS]
    ejes18 = [v18_0 + d for d in DELTAS]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, dato, titulo in ((axs[0], reach, "proxy de hits (fraccion a la caja)"),
                             (axs[1], offr, "|offset| del centroide en z=390 (mm)")):
        im = ax.imshow(dato, origin="lower", aspect="auto", cmap="viridis",
                       extent=[ejes18[0], ejes18[-1], ejes15[0], ejes15[-1]])
        ax.plot(v18_0, v15_0, "r*", ms=14, label="record actual")
        ax.set_xlabel("V18")
        ax.set_ylabel("V15")
        ax.set_title(titulo)
        ax.legend(loc="lower right", fontsize=8)
        fig.colorbar(im, ax=ax)
    fig.suptitle("Barrido de placas deflectoras (fisica dual 1mm, resto = record)")
    fig.tight_layout()
    out = ROOT / "outputs" / "scan_placas.png"
    fig.savefig(out, dpi=140)
    print(f"\nFigura: {out}")

    i = orden[0]
    print(f"\nMEJOR: {nombres[i]} -- candidato para arbitraje SIMION")


if __name__ == "__main__":
    main()
