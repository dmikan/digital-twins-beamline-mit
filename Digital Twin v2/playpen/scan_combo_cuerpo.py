"""Barrido combinado tras el hallazgo del scan de supervivencia:
cuadrupolo mas debil x V3 mas negativo. 16 vuelos SIMION.
Salida: playpen/scan_combo_cuerpo.txt
"""
import pathlib
import sys

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
    return dict(doblez=int((p[:, 2] > 150).sum()),
                plano=int((p[:, 2] > 390).sum()),
                hits=int(op.count_hits(p)))


w(f"{'escala_quad':>11} {'dV3':>6} {'supera':>7} {'plano':>6} {'hits':>5}")
mejor = None
for esc in (0.60, 0.65, 0.70, 0.75):
    for dv3 in (-300, -200, -100, 0):
        v = dict(BEST)
        v[3] += dv3
        for e in (9, 10, 11, 12):
            v[e] = BEST[e] * esc
        m = medir(v)
        w(f"{esc:>11.2f} {dv3:>+6d} {m['doblez']:>7} {m['plano']:>6} {m['hits']:>5}")
        if mejor is None or m["hits"] > mejor[2]["hits"]:
            mejor = (esc, dv3, m, dict(v))

esc, dv3, m, v = mejor
w(f"\nMEJOR: quad x{esc:.2f}, V3{dv3:+d} -> hits={m['hits']} "
  f"(supera={m['doblez']}, plano={m['plano']})")
w("voltajes: " + ", ".join(f"V{k}={vv:+.1f}" for k, vv in sorted(v.items())))

(pathlib.Path(__file__).parent / "scan_combo_cuerpo.txt").write_text(
    "\n".join(L), encoding="utf-8")
print("\nreporte: playpen/scan_combo_cuerpo.txt")
