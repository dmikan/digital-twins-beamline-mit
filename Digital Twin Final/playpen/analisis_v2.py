"""Analisis de comportamiento de la primera version del modelo v2
(objetivo J_v2 normalizado). Escribe resultados en playpen/:
  - analisis_objetivo_v2.txt   (reporte)
  - convergencia_v2.png        (J y hits por evaluacion)

Compara contra el baseline fresh50_splat (mismas metricas basadas en hits,
que son comparables entre objetivos; los valores J no lo son).
"""
import pathlib
import sys

import numpy as np
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAYPEN = ROOT / "playpen"
optuna.logging.set_verbosity(optuna.logging.WARNING)

from caracterizador import objetivo_v2  # reconstruir desgloses por trial

L = []
def w(s=""):
    print(s)
    L.append(str(s))

# ------------------------------------------------------------------ v2
st = optuna.load_study(study_name="gemelo_db_v2",
                       storage=f"sqlite:///{ROOT / 'studies' / 'gemelo_db_v2.db'}")
trials = sorted([t for t in st.trials if t.value is not None], key=lambda t: t.number)
J = np.array([t.value for t in trials])
hits = np.array([t.user_attrs.get("simion_hits", 0) for t in trials])
pred = np.array([t.user_attrs.get("rk4_score", np.nan) for t in trials])

w("=" * 72)
w("ANALISIS -- primera version del modelo v2 (objetivo J_v2 normalizado)")
w("=" * 72)
w(f"trials: {len(trials)}   mejor J_v2: {J.min():.3f}   "
  f"hitters: {(hits > 0).sum()}   mejor hits: {hits.max():g}")
primera = np.argmax(hits > 0) + 1 if (hits > 0).any() else None
w(f"primera corrida con hits: #{primera}" if primera else "sin hits")

# correlacion filtro (score de screening, mm) vs objetivo real (J_v2).
# Unidades distintas a proposito: el filtro rankea, el objetivo mide --
# lo que importa es la correlacion de RANGO.
ok = np.isfinite(pred)
if ok.sum() >= 3:
    from scipy.stats import pearsonr, spearmanr
    pr, _ = pearsonr(pred[ok], J[ok])
    sr, _ = spearmanr(pred[ok], J[ok])
    w(f"\nscore de screening RK4 (mm) vs J_v2 real (n={ok.sum()}):")
    w(f"  Pearson {pr:+.3f}   Spearman {sr:+.3f}  (positivo = el filtro ayuda)")

# desglose medio por termino (reconstruido de las features persistidas)
w("\ncontribucion media por termino (reconstruida de user_attrs):")
acum = {}
n_rec = 0
for t in trials:
    f = {k[2:]: v for k, v in t.user_attrs.items() if k.startswith("f_")}
    if not f:
        continue
    f["hits"] = t.user_attrs.get("simion_hits", 0)
    f["n_considerados"] = 500
    _, d = objetivo_v2(f, con_pared=False)
    for k, v in d.items():
        acum[k] = acum.get(k, 0.0) + v
    n_rec += 1
if n_rec:
    for k, v in sorted(acum.items(), key=lambda kv: -kv[1]):
        w(f"  {k:14s} {v / n_rec:.4f}")

# ------------------------------------------------------------------ baseline
w("\n" + "-" * 72)
w("BASELINE (fresh50_splat, primeras N corridas, metricas de hits)")
try:
    stb = optuna.load_study(
        study_name="fresh_from_scratch_50",
        storage=f"sqlite:///{ROOT / 'studies' / 'beamline_study_fresh50_splat.db'}")
    tb = sorted([t for t in stb.trials if t.value is not None],
                key=lambda t: t.number)[:len(trials)]
    hb = np.array([t.user_attrs.get("simion_hits", 0) for t in tb])
    pb = np.argmax(hb > 0) + 1 if (hb > 0).any() else None
    w(f"trials comparados: {len(tb)}   hitters: {(hb > 0).sum()}   "
      f"mejor hits: {hb.max():g}   primer hitter: #{pb}")
    w(f"\nv2 vs baseline (mismo presupuesto): hitters {(hits > 0).sum()} vs "
      f"{(hb > 0).sum()}, mejor hits {hits.max():g} vs {hb.max():g}")
except Exception as e:
    w(f"(baseline no disponible: {e})")

# ------------------------------------------------------------------ figura
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
a1.plot(range(1, len(J) + 1), np.minimum.accumulate(J), "o-")
a1.set_xlabel("evaluacion SIMION #")
a1.set_ylabel("mejor J_v2 hasta el momento")
a1.set_title("Convergencia del modelo v2")
a1.grid(alpha=0.3)
a2.bar(range(1, len(hits) + 1), hits, color="steelblue")
a2.set_xlabel("evaluacion SIMION #")
a2.set_ylabel("hits (de 500)")
a2.set_title("Hits por corrida")
a2.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(PLAYPEN / "convergencia_v2.png", dpi=140)
w(f"\nfigura: playpen/convergencia_v2.png")

(PLAYPEN / "analisis_objetivo_v2.txt").write_text("\n".join(L), encoding="utf-8")
print("\nreporte: playpen/analisis_objetivo_v2.txt")
