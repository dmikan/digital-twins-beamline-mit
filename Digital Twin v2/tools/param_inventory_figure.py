"""Genera report_figures/fig_inventario_parametros.png.
Para actualizar estados cuando un compañero termine su tarea: cambiar
"pend" -> "ok" en la tupla correspondiente y re-correr."""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "report_figures" / "fig_inventario_parametros.png"

C_OK, C_PEND, C_PURP = "#5DCAA5", "#D3D1C7", "#CECBF6"
T_OK, T_PEND, T_PURP = "#04342C", "#444441", "#26215C"

GRUPOS = [
    ("1 · Transmisión y llegada", "ya en el objetivo actual", [
        ("dist_top10 (mm)", "", "ok"), ("plane_fraction", "", "ok"),
        ("hits", "", "ok"), ("mean_splat_all", "", "ok"),
        ("wall_frac (solo RK4)", "", "ok")]),
    ("2 · Forma espacial", "grupo que llega al plano", [
        ("offset x, y (mm)", "", "ok"), ("sigma x, y (mm)", "", "ok"),
        ("halo_fraction", "A", "pend"), ("kurtosis x, y", "A", "pend")]),
    ("3 · Cinemáticos", "requieren velocidad — ya extraída", [
        ("divergencia x, y (mrad)", "", "ok"), ("twiss alpha x, y", "C→D", "pend"),
        ("twiss beta x, y", "C→D", "pend"), ("emitancia x, y", "C→D", "pend")]),
    ("4 · Aberraciones", "mayor valor diagnóstico", [
        ("residuo matriz transporte x, y", "E", "pend")]),
]

def chip(ax, x, y, w, h, label, owner, estado):
    fc, tc = (C_OK, T_OK) if estado == "ok" else (C_PEND, T_PEND)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006",
                                fc=fc, ec="none"))
    txt = label if not owner else f"{label}   ·{owner}"
    ax.text(x + 0.012, y + h / 2, txt, va="center", fontsize=9.5, color=tc)

fig, ax = plt.subplots(figsize=(11.5, 8.2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
pos = {0: (0.02, 0.92, 0.47), 1: (0.52, 0.92, 0.46),
       2: (0.02, 0.55, 0.47), 3: (0.52, 0.55, 0.46)}
ch, gap = 0.045, 0.012
for gi, (titulo, sub, params) in enumerate(GRUPOS):
    gx, y0, gw = pos[gi]
    n_rows = (len(params) + 1) // 2
    gh = 0.085 + n_rows * (ch + gap)
    ax.add_patch(plt.Rectangle((gx, y0 - gh), gw, gh, fill=False,
                               ec="#888780", lw=0.8, ls=(0, (4, 3))))
    ax.text(gx + 0.012, y0 - 0.028, titulo, fontsize=12)
    ax.text(gx + 0.012, y0 - 0.055, sub, fontsize=9, color="#5F5E5A")
    cw = (gw - 0.045) / 2
    for i, (label, owner, estado) in enumerate(params):
        wide = len(params) == 1
        cx = gx + 0.015 + (0 if (i % 2 == 0 or wide) else cw + 0.015)
        cy = y0 - 0.075 - (i // (1 if wide else 2)) * (ch + gap) - ch
        chip(ax, cx, cy, gw - 0.03 if wide else cw, ch, label, owner, estado)

chip(ax, 0.02, 0.965, 0.13, 0.03, "implementado", "", "ok")
chip(ax, 0.17, 0.965, 0.34, 0.03, "pendiente — letra = tarea del reparto", "", "pend")

fy, fh = 0.03, 0.10
boxes = [(0.02, 0.26, C_PEND, T_PEND, "~22 features", "grupos 1–4, por config"),
         (0.36, 0.28, C_PURP, T_PURP, "Ridge / Lasso — Task B", "aprende los pesos $w_i$"),
         (0.72, 0.26, C_OK, T_OK, "target  $J=\\sum w_i f_i$", "hoy: 3 términos a mano")]
for bx, bw, fc, tc, t1, t2 in boxes:
    ax.add_patch(FancyBboxPatch((bx, fy), bw, fh, boxstyle="round,pad=0.008",
                                fc=fc, ec="none"))
    ax.text(bx + 0.02, fy + fh - 0.032, t1, fontsize=12, color=tc)
    ax.text(bx + 0.02, fy + 0.026, t2, fontsize=9.5, color=tc)
for x0, x1 in ((0.285, 0.355), (0.645, 0.715)):
    ax.add_patch(FancyArrowPatch((x0, fy + fh / 2), (x1, fy + fh / 2),
                                 arrowstyle="-|>", mutation_scale=16, color="#5F5E5A"))

ax.set_title("Inventario de parámetros del analizador de beam — "
             "interfaz compartida characterize_beam()", fontsize=13, pad=14)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"ok  {OUT}")