"""
rango_confiable.py -- radio de confianza del gemelo (2026-07-07)
=================================================================

Determina hasta que DISTANCIA EN VOLTAJE de un config bueno la prediccion
del twin sigue coincidiendo con SIMION. Perturba el mejor config con ruido
gaussiano de magnitud creciente (sigma por electrodo = 25,50,100,200,400 V)
sobre los 8 electrodos optimizables, y en cada punto compara:

  - J_v2   : twin (predecir) vs SIMION (evaluar)   -> |dJ|
  - centroide (offset_x,y) : twin vs SIMION         -> |dc| (mm)
  - transmision: hits SIMION (para saber si el haz sigue vivo)

Reporta, por magnitud, la mediana de |dJ| y |dc| y define el RADIO
CONFIABLE = mayor |dV| donde el twin aun predice dentro de tolerancia.

Salida: outputs/report_figures/fig_rango_confiable.png
Correr:  python playpen/rango_confiable.py
"""

import pathlib
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import optimizer as op
from gemelo import GemeloDigital

MAGNITUDES = [0.0, 25.0, 50.0, 100.0, 200.0, 400.0]  # sigma/electrodo (V)
N_MUESTRAS = 3
TOL_J = 0.10          # |dJ| aceptable
TOL_C = 2.0           # |dc| aceptable (mm)
OPTIM = sorted(op.OPTIMIZE)


def main():
    # mejor config por hits entre los dos estudios
    best = None
    for est, db in (("gemelo_db_v2", "studies/gemelo_db_v2.db"),
                    ("gemelo_db_v2", "studies/gemelo_db_v2.db")):
        tw0 = GemeloDigital(estudio=est, db=ROOT / db)
        m = tw0.mejor(por="hits")
        if m and (best is None or (m["hits"] or 0) > (best[1]["hits"] or 0)):
            best = (tw0, m)
    tw, m = best
    base = tw.voltajes_completos(m["voltajes"])
    print(f"Config base ({m['hits']} hits): perturbando {len(OPTIM)} electrodos optimizables\n")

    rng = np.random.default_rng(7)
    filas = []  # (mag, dv_l2, dJ, dc, hits, twin_reach)
    for mag in MAGNITUDES:
        reps = 1 if mag == 0.0 else N_MUESTRAS
        for _ in range(reps):
            v = base.copy()
            if mag > 0:
                for e in OPTIM:
                    lo, hi = op.OPTIMIZE[e]
                    v[e - 1] = np.clip(base[e - 1] + rng.normal(0, mag), lo, hi)
            dv_l2 = float(np.sqrt(sum((v[e - 1] - base[e - 1]) ** 2 for e in OPTIM)))

            p = tw.predecir(v)            # twin
            r = tw.evaluar(v)             # SIMION
            fp, fr = p["features"], (r.get("features") or {})
            dJ = abs(p["objetivo"] - r["objetivo"])
            cx_t, cy_t = fp.get("offset_x_mm", np.nan), fp.get("offset_y_mm", np.nan)
            cx_s, cy_s = fr.get("offset_x_mm", np.nan), fr.get("offset_y_mm", np.nan)
            dc = float(np.hypot(cx_t - cx_s, cy_t - cy_s))
            filas.append((mag, dv_l2, dJ, dc, r["hits"], p["reach_fraction"]))
            print(f"  mag={mag:>5.0f}V |dV|={dv_l2:>4.0f}  |dJ|={dJ:.3f}  "
                  f"|dc|={dc if np.isfinite(dc) else float('nan'):>5.1f}mm  "
                  f"hits={r['hits']:>3}  twin_reach={p['reach_fraction']:.2f}")

    filas = np.array(filas, dtype=float)
    print("\n===== RESUMEN POR MAGNITUD (mediana) =====")
    print(f"  {'sigma/elec':>10} {'|dV|_L2':>8} {'med|dJ|':>8} {'med|dc|mm':>10} {'hits':>6}")
    radio_J = radio_C = 0.0
    for mag in MAGNITUDES:
        sub = filas[filas[:, 0] == mag]
        mdJ = np.nanmedian(sub[:, 2])
        mdc = np.nanmedian(sub[:, 3])
        mdv = np.nanmedian(sub[:, 1])
        mh = np.nanmedian(sub[:, 4])
        print(f"  {mag:>10.0f} {mdv:>8.0f} {mdJ:>8.3f} {mdc:>10.1f} {mh:>6.0f}")
        if mdJ <= TOL_J:
            radio_J = mdv
        if np.isfinite(mdc) and mdc <= TOL_C:
            radio_C = mdv

    print(f"\n  RADIO CONFIABLE (|dV|_L2 desde el config base):")
    print(f"    objetivo J_v2 dentro de {TOL_J}: hasta ~{radio_J:.0f} V")
    print(f"    centroide dentro de {TOL_C}mm : hasta ~{radio_C:.0f} V")

    # figura
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.axhline(TOL_J, color="r", ls="--", lw=1, label=f"tol {TOL_J}")
    a1.scatter(filas[:, 1], filas[:, 2], c="tab:blue", alpha=0.7)
    a1.set_xlabel("|dV| desde base (L2, V)"); a1.set_ylabel("|J_twin - J_SIMION|")
    a1.set_title("Fidelidad del objetivo vs distancia"); a1.legend(); a1.grid(alpha=0.3)
    finite = np.isfinite(filas[:, 3])
    a2.axhline(TOL_C, color="r", ls="--", lw=1, label=f"tol {TOL_C}mm")
    a2.scatter(filas[finite, 1], filas[finite, 3], c="tab:green", alpha=0.7)
    a2.set_xlabel("|dV| desde base (L2, V)"); a2.set_ylabel("|centroide_twin - centroide_SIMION| (mm)")
    a2.set_title("Fidelidad del centroide vs distancia"); a2.legend(); a2.grid(alpha=0.3)
    fig.suptitle(f"Radio de confianza del gemelo (base = mejor config, {m['hits']} hits)")
    fig.tight_layout()
    out = ROOT / "outputs" / "report_figures" / "fig_rango_confiable.png"
    fig.savefig(out, dpi=140)
    print(f"\nFigura: {out}")


if __name__ == "__main__":
    main()
