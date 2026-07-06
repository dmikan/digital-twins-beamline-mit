"""Diagnostico: responde la SUPERVIVENCIA DEL CUERPO del haz a los mandos
de entrada? (V3 = einzel 1, s = escala del cuadrupolo). Medimos con splats
reales de SIMION, no hits:
  n_supera_doblez : iones con z de splat > 150 mm (salieron del codo)
  n_plane         : iones con z > 390 mm (llegaron al plano del detector)
  hits            : ventana estricta (referencia)
~13 vuelos SIMION (~80 s). Salida: playpen/scan_supervivencia.txt
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import optimizer as op
from orchestrator import apply_voltages, run_simion, FLY_COMMAND

BEST = {3: -269.6, 6: -208.0, 9: -443.8, 10: 336.1, 11: 372.0, 12: -322.3,
        15: -0.8, 18: -252.6}

L = []
def w(s=""):
    print(s)
    L.append(str(s))


def medir(volt):
    apply_voltages(volt)
    p = op.get_positions(run_simion(FLY_COMMAND))
    hits = int(op.count_hits(p))
    return dict(n=len(p), doblez=int((p[:, 2] > 150).sum()),
                plano=int((p[:, 2] > 390).sum()), hits=hits)


w(f"{'config':<22} {'supera doblez':>13} {'llega plano':>11} {'hits':>5}")
m = medir(BEST)
w(f"{'base (mejor)':<22} {m['doblez']:>10}/{m['n']} {m['plano']:>11} {m['hits']:>5}")

for dv3 in (-300, -200, -100, 100, 200, 300):
    v = dict(BEST)
    v[3] += dv3
    m = medir(v)
    w(f"{'V3 %+d' % dv3:<22} {m['doblez']:>10}/500 {m['plano']:>11} {m['hits']:>5}")

for esc in (0.7, 0.85, 1.15, 1.3):
    v = dict(BEST)
    for e in (9, 10, 11, 12):
        v[e] = BEST[e] * esc
    m = medir(v)
    w(f"{'cuadrupolo x%.2f' % esc:<22} {m['doblez']:>10}/500 {m['plano']:>11} {m['hits']:>5}")

# combinacion: mejor enfoque de entrada conocido del steer_scan (V6+15)
v = dict(BEST)
v[6] += 15
m = medir(v)
w(f"{'V6 +15 (steer_scan)':<22} {m['doblez']:>10}/500 {m['plano']:>11} {m['hits']:>5}")

(pathlib.Path(__file__).parent / "scan_supervivencia.txt").write_text(
    "\n".join(L), encoding="utf-8")
print("\nreporte: playpen/scan_supervivencia.txt")
