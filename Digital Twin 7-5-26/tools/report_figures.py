"""
report_figures.py
==================

Genera las figuras del informe a partir de los estudios persistidos
(ninguna corrida RK4 nueva; la figura (d) hace UNA corrida SIMION real).

  (a) fig_cobertura_espacio.png   Cobertura del espacio de voltajes
                                  explorado (item 2 del informe).
  (b) fig_predicho_vs_real.png    Prediccion del gemelo RK4 vs resultado
                                  SIMION real, por trial (item 4).
  (c) fig_convergencia.png        Mejor resultado vs numero de evaluacion
                                  SIMION, por estudio (items 4/5).
  (d) fig_haz_en_detector.png     Donde muere el haz + forma del haz en el
      splats_mejor_config.csv     plano del detector, con los splats REALES
                                  de una corrida SIMION del mejor config
                                  (items 4/5/7). Requiere SIMION: se salta
                                  con --sin-simion.

Salida en report_figures/.  Correr con:
    python report_figures.py [--sin-simion]
"""

import glob
import pathlib
import sys

import numpy as np
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import optimizer as op

OUT = ROOT / "outputs" / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)

BEST_PARAMS = None  # set in collect(), used by (d)


def all_studies():
    dbs = sorted(glob.glob(str(ROOT / "studies" / "beamline_study_*.db"))) + \
          sorted(glob.glob(str(ROOT / "legacy" / "studies" / "beamline_study_*.db")))
    for db in dbs:
        for s in optuna.study.get_all_study_summaries(storage=f"sqlite:///{db}"):
            study = optuna.load_study(study_name=s.study_name, storage=f"sqlite:///{db}")
            yield pathlib.Path(db).stem.replace("beamline_study_", ""), study


def collect():
    """Every archived SIMION trial: params, hits, value, direction, rk4 pred."""
    global BEST_PARAMS
    rows = []
    best_hits = -1
    for label, study in all_studies():
        maximize = study.direction == optuna.study.StudyDirection.MAXIMIZE
        for t in study.trials:
            if t.value is None or not all(f"V{e}" in t.params for e in op.OPTIMIZE):
                continue
            hits = t.user_attrs.get("simion_hits")
            if hits is None:
                hits = t.value if (maximize and t.value >= 0) else None
            rows.append(dict(
                label=label, number=t.number, params=t.params, value=t.value,
                maximize=maximize, hits=hits,
                rk4=t.user_attrs.get("rk4_score"),
            ))
            if hits is not None and hits > best_hits:
                best_hits, BEST_PARAMS = hits, t.params
    print(f"{len(rows)} trials SIMION registrados; mejor: {best_hits:g} hits")
    return rows


def fig_cobertura(rows):
    pairs = [(9, 10), (11, 12), (3, 6), (15, 18)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (ea, eb) in zip(axes.ravel(), pairs):
        xs = np.array([r["params"][f"V{ea}"] for r in rows])
        ys = np.array([r["params"][f"V{eb}"] for r in rows])
        hits = np.array([r["hits"] if r["hits"] is not None else 0 for r in rows])
        miss = hits == 0
        ax.scatter(xs[miss], ys[miss], s=8, c="0.75", label="0 hits")
        if (~miss).any():
            sc = ax.scatter(xs[~miss], ys[~miss], s=40, c=hits[~miss],
                            cmap="viridis", vmin=1, label="con hits")
            plt.colorbar(sc, ax=ax, label="hits (de 500)")
        ax.set_xlabel(f"V{ea} (V)")
        ax.set_ylabel(f"V{eb} (V)")
        ax.set_xlim(-1050, 1050)
        ax.set_ylim(-1050, 1050)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Cobertura del espacio de voltajes: {len(rows)} evaluaciones SIMION\n"
                 "(gris = 0 hits; color = iones en el detector)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_cobertura_espacio.png", dpi=150)
    plt.close(fig)
    print("ok  fig_cobertura_espacio.png")


def fig_predicho_vs_real(rows):
    # Solo trials con prediccion RK4 en mm (posteriores a la alineacion del
    # score): rk4_score > 5 distingue la escala mm de la escala vieja (~0.1).
    pts = [r for r in rows if not r["maximize"] and r["rk4"] is not None and r["rk4"] > 5]
    if not pts:
        print("--  sin datos para fig_predicho_vs_real")
        return
    x = np.array([r["rk4"] for r in pts])
    y = np.array([r["value"] for r in pts])
    hits = np.array([r["hits"] or 0 for r in pts])

    fig, ax = plt.subplots(figsize=(7.5, 6))
    miss = hits == 0
    ax.scatter(x[miss], y[miss], s=25, c="0.6", label="0 hits")
    if (~miss).any():
        sc = ax.scatter(x[~miss], y[~miss], s=70, c=hits[~miss], cmap="viridis",
                        vmin=1, label="con hits")
        plt.colorbar(sc, ax=ax, label="hits (de 500)")
    lim = [0, max(x.max(), y.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=0.8, label="prediccion perfecta")
    ax.set_xlabel("Prediccion del gemelo RK4 (mm, menor = mejor)")
    ax.set_ylabel("Resultado SIMION real (mm)")
    ax.set_title(f"Predicho vs real por trial (n={len(pts)})\n"
                 "Validacion offline sobre 288 configs: Spearman +0.458,\n"
                 "17/61 hitters en el top-30 del filtro", fontsize=10)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_predicho_vs_real.png", dpi=150)
    plt.close(fig)
    print(f"ok  fig_predicho_vs_real.png  ({len(pts)} trials)")


def fig_convergencia(rows):
    by_study = {}
    for r in rows:
        by_study.setdefault((r["label"], r["maximize"]), []).append(r)

    fig, (ax_h, ax_d) = plt.subplots(1, 2, figsize=(12, 5))
    for (label, maximize), rs in sorted(by_study.items()):
        rs = sorted(rs, key=lambda r: r["number"])
        vals = [r["value"] for r in rs]
        if maximize:
            running = np.maximum.accumulate(vals)
            ax_h.plot(range(1, len(vals) + 1), running, marker=".", label=label)
        else:
            running = np.minimum.accumulate(vals)
            ax_d.plot(range(1, len(vals) + 1), running, marker=".", label=label)
    ax_h.set_title("Estudios con objetivo 'hits' (maximizar)")
    ax_h.set_xlabel("evaluacion SIMION #")
    ax_h.set_ylabel("mejor hits hasta el momento")
    ax_d.set_title("Estudios con objetivo denso (minimizar mm)")
    ax_d.set_xlabel("evaluacion SIMION #")
    ax_d.set_ylabel("mejor distancia (mm)")
    for ax in (ax_h, ax_d):
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Convergencia: mejor resultado vs presupuesto SIMION gastado")
    fig.tight_layout()
    fig.savefig(OUT / "fig_convergencia.png", dpi=150)
    plt.close(fig)
    print("ok  fig_convergencia.png")


def fig_haz_detector():
    """UNA corrida SIMION real del mejor config, guardando los splats."""
    from orchestrator import apply_voltages, run_simion, FLY_COMMAND
    chosen = {int(k.lstrip("V")): float(v) for k, v in BEST_PARAMS.items()}
    print(f"Volando el mejor config en SIMION: { {k: round(v) for k, v in sorted(chosen.items())} }")
    apply_voltages(chosen)
    out = run_simion(FLY_COMMAND)
    positions = op.get_positions(out)
    hits = op.count_hits(positions)
    np.savetxt(OUT / "splats_mejor_config.csv", positions, delimiter=",",
               header="x_mm,y_mm,z_mm", comments="")
    print(f"    {positions.shape[0]} splats registrados, {hits} hits")

    xr, yr, zr = op.DETECTOR_REGION["x"], op.DETECTOR_REGION["y"], op.DETECTOR_REGION["z"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(positions[:, 2], bins=60, color="steelblue")
    ax1.axvspan(zr[0], zr[1], color="tab:green", alpha=0.4,
                label=f"detector z={zr[0]}-{zr[1]}")
    ax1.set_xlabel("z del splat (mm)")
    ax1.set_ylabel("iones")
    ax1.set_title("Donde termina cada ion (500 iones)")
    ax1.legend()

    near = positions[positions[:, 2] > 350]
    ax2.scatter(near[:, 0], near[:, 1], s=18, c=near[:, 2], cmap="plasma")
    ax2.add_patch(plt.Rectangle((xr[0], yr[0]), xr[1] - xr[0], yr[1] - yr[0],
                                fill=False, edgecolor="tab:green", lw=2,
                                label="ventana del detector"))
    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel("y (mm)")
    ax2.set_title(f"Splats con z>350mm (n={len(near)}) -- hits={hits}")
    ax2.legend()
    ax2.set_aspect("equal")

    fig.suptitle("Forma real del haz (SIMION) para el mejor config encontrado")
    fig.tight_layout()
    fig.savefig(OUT / "fig_haz_en_detector.png", dpi=150)
    plt.close(fig)
    print("ok  fig_haz_en_detector.png + splats_mejor_config.csv")


if __name__ == "__main__":
    rows = collect()
    fig_cobertura(rows)
    fig_predicho_vs_real(rows)
    fig_convergencia(rows)
    if "--sin-simion" not in sys.argv:
        fig_haz_detector()
    print(f"\nFiguras en: {OUT}")
