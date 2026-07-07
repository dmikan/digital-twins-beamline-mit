"""
validar_objetivo_v22.py -- prueba del fix de transmision empinada
==================================================================

Recomputa J_v2 (formula ACTUAL de caracterizador) sobre las features
guardadas de las 47 corridas SIMION del registro y verifica que el bug
quedo resuelto:

  1. el record (mas hits) queda #1 (menor J)
  2. hits y J estan anticorrelados (mas hits -> menor J): Spearman muy
     negativo, y NINGUN config de 0 hits le gana a uno de >=10 hits.

Compara contra la formula VIEJA (transmision lineal, pesos v2.1) para
mostrar el antes/despues.

Correr:  python playpen/validar_objetivo_v22.py
"""

import json
import pathlib
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import caracterizador as C


def j_viejo(f):
    """La formula v2.1 (transmision lineal, pesos viejos) para el antes."""
    pesos = dict(transmision=0.40, cuerpo=0.12, acercamiento=0.13, offset=0.10,
                 halo=0.08, kurtosis=0.06, colimacion=0.06, twiss=0.05)
    n = max(1, f.get("n_considerados") or f.get("n_total") or 1)
    hits = f.get("hits") or 0
    off_x, off_y = f.get("offset_x_mm", np.nan), f.get("offset_y_mm", np.nan)
    radio = (np.hypot(off_x / C._MEDIA_VENTANA[0], off_y / C._MEDIA_VENTANA[1])
             if np.isfinite(off_x) and np.isfinite(off_y) else np.nan)
    alfa = np.nanmean([abs(f.get("twiss_alpha_x", np.nan)), abs(f.get("twiss_alpha_y", np.nan))])
    div = np.nanmean([f.get("div_x_mrad", np.nan), f.get("div_y_mrad", np.nan)])
    kurt = np.nanmax([f.get("kurtosis_x", np.nan), f.get("kurtosis_y", np.nan)])
    halo = f.get("halo_fraction", np.nan)
    pf = f.get("plane_fraction")
    t = dict(transmision=1.0 - hits / n,
             cuerpo=1.0 - float(pf) if pf is not None and np.isfinite(pf) else 1.0,
             acercamiento=C._racional(f.get("dist_punta_mm", np.nan), C.D0_MM),
             offset=C._racional(radio, 1.0),
             halo=float(halo) if np.isfinite(halo) else 1.0,
             kurtosis=C._racional(max(kurt, 0.0) if np.isfinite(kurt) else np.nan, C.K0_KURT),
             colimacion=C._racional(div, C.DIV0_MRAD),
             twiss=C._racional(alfa ** 2 if np.isfinite(alfa) else np.nan, 1.0))
    return sum(pesos[k] * v for k, v in t.items())


def main():
    reg = ROOT / "studies" / "registro_corridas.jsonl"
    filas = [json.loads(l) for l in reg.read_text().splitlines() if l.strip()]
    sim = [f for f in filas if f.get("fuente") == "simion"
           and f.get("hits") is not None and f.get("features")]
    print(f"{len(sim)} corridas SIMION con features en el registro\n")

    hits = np.array([f["hits"] for f in sim])
    j_new = np.array([C.objetivo_v2(f["features"], con_pared=False)[0] for f in sim])
    j_old = np.array([j_viejo(f["features"]) for f in sim])

    for nombre, j in (("VIEJA (v2.1)", j_old), ("NUEVA (v2.2)", j_new)):
        sr, _ = spearmanr(hits, j)
        idx_best_j = int(np.argmin(j))
        # cuantos 0-hit le ganan (J menor) a algun config de >=10 hits
        j_10 = j[hits >= 10]
        peor_10 = j_10.max() if len(j_10) else np.inf
        cero_ganan = int(((hits == 0) & (j < peor_10)).sum())
        print(f"  {nombre}: Spearman(hits,J)={sr:+.3f}  "
              f"| mejor J tiene {hits[idx_best_j]} hits  "
              f"| {cero_ganan} configs de 0 hits rankean sobre el peor de >=10 hits")

    print("\n  (queremos: Spearman MUY negativo, mejor J = mas hits, 0 configs "
          "de 0 hits ganando)")

    orden = np.argsort(j_new)
    print("\n  TOP 6 por J v2.2 (nuevo):")
    for i in orden[:6]:
        print(f"    J={j_new[i]:.3f}  hits={hits[i]:>3}")


if __name__ == "__main__":
    main()
