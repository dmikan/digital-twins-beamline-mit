"""
steer_scan.py
==============

Paso 1 del plan post-deadline: caracterizar y comprimir/centrar el grupo
de iones que YA llega al plano del detector con el mejor config conocido.

Hallazgo que motiva este barrido (medido sobre los splats reales,
report_figures/splats_mejor_config.csv): el grupo que llega esta casi
centrado (dx=+0.4mm, dy=+1.5mm respecto del centro de la ventana), pero es
demasiado ANCHO: RMS 8.4mm en x y 15.5mm en y contra una ventana de
+-6.0 x +-6.5mm. Centrarlo rigidamente no ganaria nada (seguirian entrando
~4); hay que COMPRIMIRLO. Por eso el barrido prioriza los mandos de
enfoque (V6, asimetrias del cuadrupolo) sobre los de direccion (V15, V18).

Metodo: barridos 1-D alrededor del mejor config conocido, evaluados
DIRECTAMENTE en SIMION (sin RK4: las diferencias de milimetros que
buscamos estan por debajo de la resolucion de ranking fino del gemelo,
medida en la validacion). Por cada corrida se registra la caracterizacion
completa del grupo que llega, no solo los hits.

Mandos barridos (delta respecto del mejor config):
  V6     lente einzel 2                     (enfoque)
  asymA  V10 += d, V11 -= d                 (asimetria del par +)
  asymB  V9  += d, V12 -= d                 (asimetria del par -)
  V15    placa deflectora 1                 (direccion)
  V18    placa deflectora 2                 (direccion)
  V3     lente einzel 1                     (enfoque aguas arriba)

Salidas: steer_scan_results.csv (una fila por corrida SIMION) y resumen en
consola. Corridas SIMION: len(DELTAS) x 6 mandos + 1 base (~31 x ~5.5s).

Correr con:
    python steer_scan.py
"""

import csv
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import optimizer as op
from orchestrator import apply_voltages, run_simion, FLY_COMMAND

OUT_CSV = ROOT / "outputs" / "steer_scan_results.csv"

# Mejor config conocido (trial con 10 hits; identico al best de todos los
# estudios -- ver SESSION_HANDOFF).
BEST = {3: -269.6, 6: -208.0, 9: -443.8, 10: 336.1, 11: 372.0, 12: -322.3,
        15: -0.8, 18: -252.6}

DELTAS = [-80.0, -40.0, -15.0, 15.0, 40.0, 80.0]

XW, YW, ZW = op.DETECTOR_REGION["x"], op.DETECTOR_REGION["y"], op.DETECTOR_REGION["z"]
CX, CY = (XW[0] + XW[1]) / 2, (YW[0] + YW[1]) / 2


def knob_apply(base, knob, d):
    v = dict(base)
    if knob == "asymA":
        v[10] += d
        v[11] -= d
    elif knob == "asymB":
        v[9] += d
        v[12] -= d
    else:
        v[int(knob.lstrip("V"))] += d
    return v


def fly(voltages):
    apply_voltages(voltages)
    out = run_simion(FLY_COMMAND)
    return op.get_positions(out)


def characterize(p):
    hits = int(op.count_hits(p))
    near = p[p[:, 2] > 390]
    row = dict(hits=hits, n_plane=len(near))
    if len(near) >= 3:
        row.update(
            cx=float(near[:, 0].mean()), cy=float(near[:, 1].mean()),
            dx=float(near[:, 0].mean() - CX), dy=float(near[:, 1].mean() - CY),
            rms_x=float(near[:, 0].std()), rms_y=float(near[:, 1].std()),
            n_z_window=int(((near[:, 2] > ZW[0]) & (near[:, 2] < ZW[1])).sum()),
        )
    else:
        row.update(cx=np.nan, cy=np.nan, dx=np.nan, dy=np.nan,
                   rms_x=np.nan, rms_y=np.nan, n_z_window=0)
    return row


def main():
    op.check_setup()
    results = []

    print("Base (mejor config conocido)...")
    t0 = time.time()
    row = dict(knob="base", delta=0.0, **characterize(fly(BEST)))
    print(f"  base: hits={row['hits']} n_plane={row['n_plane']} "
          f"rms=({row['rms_x']:.1f},{row['rms_y']:.1f})  ({time.time()-t0:.1f}s)")
    results.append(row)

    for knob in ("V6", "asymA", "asymB", "V15", "V18", "V3"):
        for d in DELTAS:
            t0 = time.time()
            p = fly(knob_apply(BEST, knob, d))
            row = dict(knob=knob, delta=d, **characterize(p))
            results.append(row)
            print(f"  {knob:6s} {d:+6.0f}V: hits={row['hits']:3d} n_plane={row['n_plane']:3d} "
                  f"dx={row['dx']:+6.2f} dy={row['dy']:+6.2f} "
                  f"rms=({row['rms_x']:5.2f},{row['rms_y']:5.2f}) "
                  f"({time.time()-t0:.1f}s)")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nGuardado: {OUT_CSV.name}")

    base_hits = results[0]["hits"]
    best_row = max(results, key=lambda r: r["hits"])
    print(f"\nMejor corrida del barrido: {best_row['knob']} {best_row['delta']:+.0f}V "
          f"-> hits={best_row['hits']} (base={base_hits})")
    print("Top 5 por hits:")
    for r in sorted(results, key=lambda r: -r["hits"])[:5]:
        print(f"  {r['knob']:6s} {r['delta']:+6.0f}V  hits={r['hits']:3d} "
              f"n_plane={r['n_plane']:3d} rms_y={r['rms_y']:.2f}")


if __name__ == "__main__":
    main()
