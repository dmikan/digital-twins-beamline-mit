"""
report_figures.py
==================

Genera las figuras del reporte a partir de los estudios persistidos
y la bitácora de corridas del facade.

  (a) fig_solucion_quad_3d.png           Espacio de soluciones del bender en 3D (Item 4).
  (b) fig_validacion_gemelo.png          Predicción gemelo vs SIMION en la región informativa (Item 4).
  (c) fig_incertidumbre_extrapolacion.png Incertidumbre del GP y extrapolación en 1D (Item 4).
  (d) fig_control_consigna_inversa.png    Placas de deflexión y ajuste de consigna inversa (Item 5).
  (e) fig_convergencia_gemelo.png        Trayectorias de convergencia TPE vs GP (Item 5).
  (f) fig_haz_en_detector.png            Plano del detector final splats (opcional con SIMION).

Salida en outputs/report_figures/. Correr con:
    python report_figures.py [--sin-simion]
"""

import json
import glob
import pathlib
import sys
import numpy as np
import optuna
from scipy.stats import pearsonr, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import optimizer as op

OUT = ROOT / "outputs" / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)

BEST_PARAMS = None  # set in collect(), used by (f)


# Los dos estudios del reporte: TPE (gemelo_v2) y GP-seeded (gemelo_db_v2).
STUDIES_REPORTE = ("gemelo_v2", "gemelo_db_v2")


def all_studies():
    """Rinde (nombre_estudio, study) para cada estudio del reporte,
    etiquetando por NOMBRE DE ESTUDIO (no por archivo .db) y quedandose
    con la instancia mas grande de cada uno -- gemelo_v2.db contiene por
    accidente 54 trials de gemelo_db_v2 ademas de los de gemelo_v2, y
    etiquetar por archivo los mezclaba."""
    best = {}  # study_name -> (n_trials, study)
    for db in sorted(glob.glob(str(ROOT / "studies" / "*.db"))):
        try:
            summaries = optuna.study.get_all_study_summaries(storage=f"sqlite:///{db}")
        except Exception:
            continue
        for s in summaries:
            if s.study_name not in STUDIES_REPORTE:
                continue
            study = optuna.load_study(study_name=s.study_name, storage=f"sqlite:///{db}")
            n = sum(1 for t in study.trials if t.value is not None)
            if s.study_name not in best or n > best[s.study_name][0]:
                best[s.study_name] = (n, study)
    for name in STUDIES_REPORTE:
        if name in best:
            print(f"  estudio {name}: {best[name][0]} trials")
            yield name, best[name][1]


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
                offset_x=t.user_attrs.get("f_offset_x_mm"),
                offset_y=t.user_attrs.get("f_offset_y_mm"),
            ))
            if hits is not None and hits > best_hits:
                best_hits, BEST_PARAMS = hits, t.params
    print(f"{len(rows)} trials SIMION registrados; mejor: {best_hits:g} hits")
    return rows


def fig_solucion_quad_3d(rows):
    # Filter rows with hits
    pts = [r for r in rows if r["hits"] is not None]
    if not pts:
        print("--  sin datos para fig_solucion_quad_3d")
        return
    
    v9 = np.array([r["params"]["V9"] for r in pts])
    v10 = np.array([r["params"]["V10"] for r in pts])
    v12 = np.array([r["params"]["V12"] for r in pts])
    hits = np.array([r["hits"] for r in pts])
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 0 hits in light gray
    miss = hits == 0
    ax.scatter(v9[miss], v10[miss], v12[miss], s=12, c="0.8", alpha=0.4, label="0 hits")
    
    # > 0 hits in color
    if (~miss).any():
        sc = ax.scatter(v9[~miss], v10[~miss], v12[~miss], s=40, c=hits[~miss],
                        cmap="viridis", edgecolor='k', lw=0.3, vmin=1, label="con hits")
        fig.colorbar(sc, ax=ax, label="SIMION Hits (de 500)", shrink=0.6, aspect=10)
        
    # Fit 3D Gaussian to successful trials (hits > 20)
    good = hits > 20
    if good.sum() > 5:
        X_good = np.column_stack([v9[good], v10[good], v12[good]])
        mean = np.mean(X_good, axis=0)
        cov = np.cov(X_good, rowvar=False)
        
        # Calculate eigenvalues and eigenvectors of covariance matrix
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        # Generate 3D sphere grid
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones_like(u), np.cos(v))
        
        # Scale sphere to 2-sigma radii (95% confidence volume)
        radii = 2.0 * np.sqrt(np.maximum(eigenvals, 1e-9))
        
        # Transform sphere into rotated/shifted 3D covariance ellipsoid
        ellipsoid = np.zeros((30, 30, 3))
        for i in range(30):
            for j in range(30):
                point = np.array([x_sphere[i, j] * radii[0],
                                  y_sphere[i, j] * radii[1],
                                  z_sphere[i, j] * radii[2]])
                ellipsoid[i, j, :] = eigenvecs.dot(point) + mean
                
        # Draw translucent red wireframe ellipsoid representing the 3D Gaussian
        ax.plot_wireframe(ellipsoid[:, :, 0], ellipsoid[:, :, 1], ellipsoid[:, :, 2],
                          color="red", alpha=0.15, linewidth=0.5, label=r"Elipsoide de Covarianza ($2\sigma$)")
        
    ax.set_xlabel("Voltaje V9 (V)")
    ax.set_ylabel("Voltaje V10 (V)")
    ax.set_zlabel("Voltaje V12 (V)")
    ax.set_title("Espacio de Soluciones del Cuadrupolo (3D) con Ajuste Gaussiano\n"
                 "El elipsoide rojo representa el volumen de covarianza de 2 desviaciones estándar")
    ax.legend(loc="upper left")
    
    # Set standard viewing angle
    ax.view_init(elev=25, azim=45)
    
    fig.tight_layout()
    fig.savefig(OUT / "fig_solucion_quad_3d.png", dpi=150)
    plt.close(fig)
    print("ok  fig_solucion_quad_3d.png")


def fig_validacion_gemelo(rows):
    # Only trials with localized RK4 scores (in mm)
    pts = [r for r in rows if not r["maximize"] and r["rk4"] is not None and r["rk4"] > 5]
    if not pts:
        print("--  sin datos para fig_validacion_gemelo")
        return
    
    # Filter for the informative region: SIMION real cost value < 0.6 mm
    pts_inf = [r for r in pts if r["value"] < 0.6]
    if not pts_inf:
        pts_inf = pts
        
    x = np.array([r["rk4"] for r in pts_inf])
    y = np.array([r["value"] for r in pts_inf])
    hits = np.array([r["hits"] or 0 for r in pts_inf])
    
    pr, _ = pearsonr(x, y) if len(x) > 2 else (0.0, 0.0)
    sr, _ = spearmanr(x, y) if len(x) > 2 else (0.0, 0.0)
    
    fig, ax = plt.subplots(figsize=(8, 6.5))
    miss = hits == 0
    ax.scatter(x[miss], y[miss], s=35, c="0.6", alpha=0.5, label="0 hits")
    if (~miss).any():
        sc = ax.scatter(x[~miss], y[~miss], s=75, c=hits[~miss], cmap="viridis",
                        edgecolor='k', lw=0.4, vmin=1, label="con hits", zorder=3)
        plt.colorbar(sc, ax=ax, label="SIMION Hits")
        
    ax.set_xlabel("Predicción del Gemelo RK4 (mm, menor = mejor)")
    ax.set_ylabel("Resultado SIMION real (mm)")
    ax.set_title(f"Validación del Gemelo en Región Informativa (n={len(pts_inf)})\n"
                 f"Pearson: {pr:+.3f}, Spearman: {sr:+.3f}", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.25)
    
    # Autoscale X and Y axes independently to expand the data cloud
    ax.set_xlim(x.min() - 2, x.max() + 2)
    ax.set_ylim(y.min() - 0.02, y.max() + 0.02)
    
    fig.tight_layout()
    fig.savefig(OUT / "fig_validacion_gemelo.png", dpi=150)
    plt.close(fig)
    print(f"ok  fig_validacion_gemelo.png (n={len(pts_inf)})")


def numpy_gp_predict_8d(X_train, y_train, X_test, length_scale=300.0, sigma_f=0.5, noise=0.08):
    """Pure NumPy 8D Gaussian Process Regression using RBF kernel."""
    y_mean = np.mean(y_train)
    y_norm = y_train - y_mean
    
    # Kernel helper in 8D
    def kernel(x1, x2):
        sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)
        
    K = kernel(X_train, X_train) + (noise**2) * np.eye(len(X_train))
    K_inv = np.linalg.inv(K)
    
    Ks = kernel(X_test, X_train)
    Kss = kernel(X_test, X_test)
    
    y_pred = Ks.dot(K_inv).dot(y_norm) + y_mean
    var = np.diag(Kss) - np.sum(Ks.dot(K_inv) * Ks, axis=1)
    std = np.sqrt(np.maximum(var, 0.0))
    return y_pred, std


def generate_gp_sweep(X_train, y_train, hits, cols, var_name):
    col_idx = cols.index(int(var_name.lstrip("V")))
    
    x_plot_1d = np.linspace(-1000, 1000, 500)
    X_test = np.zeros((500, len(cols)))
    for i, e in enumerate(cols):
        X_test[:, i] = BEST_PARAMS[f"V{e}"]
    X_test[:, col_idx] = x_plot_1d
    
    # Quadrupoles are much more sensitive, so we use a smaller length-scale (100) compared to lenses (380)
    ls = 100.0 if var_name == "V9" else 380.0
    
    y_pred, sigma = numpy_gp_predict_8d(X_train, y_train, X_test, length_scale=ls, sigma_f=0.5, noise=0.08)
    
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot(x_plot_1d, y_pred, 'b-', lw=1.8, label=rf"Corte de Predicción GP ($\mu$ en óptimo)")
    ax.fill_between(x_plot_1d, y_pred - 2*sigma, y_pred + 2*sigma,
                    color='blue', alpha=0.15, label=r"Incertidumbre GP ($\pm 2\sigma$)")
    
    # Observations projected on this specific dimension
    sc = ax.scatter(X_train[:, col_idx], y_train, c=hits, cmap="viridis", s=45,
                    edgecolor='k', lw=0.5, label="Observaciones (Proyección de nube 8D)", zorder=3)
    plt.colorbar(sc, ax=ax, label="SIMION Hits")
    
    ax.set_xlabel(f"Voltaje Electrodo {var_name} (V)")
    ax.set_ylabel("Costo del objetivo J_v2 (mm, menor = mejor)")
    ax.set_title(f"Incertidumbre y Extrapolación del GP (Corte 1D de {var_name})\n"
                 f"La incertidumbre GP se contrae cerca del óptimo ({var_name} ~ {BEST_PARAMS[var_name]:.1f}V)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    ax.set_ylim(-0.1, 1.25)
    
    fig.tight_layout()
    filename = f"fig_incertidumbre_{var_name}.png"
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)
    print(f"ok  {filename}")


def fig_incertidumbre_extrapolacion(rows):
    pts = [r for r in rows if r["hits"] is not None]
    if len(pts) < 10:
        print("--  insuficientes datos para fig_incertidumbre_extrapolacion")
        return
    
    # Feature matrix: shapes (N, 8) corresponding to op.OPTIMIZE
    cols = sorted(op.OPTIMIZE)
    X_train = np.array([[r["params"][f"V{e}"] for e in cols] for r in pts])
    y_train = np.array([r["value"] for r in pts])
    hits = np.array([r["hits"] for r in pts])
    
    # Generate sweeps for V3 (Einzel 1), V6 (Einzel 2), and V9 (Bender 1)
    generate_gp_sweep(X_train, y_train, hits, cols, "V3")
    generate_gp_sweep(X_train, y_train, hits, cols, "V6")
    generate_gp_sweep(X_train, y_train, hits, cols, "V9")


def simple_linear_fit(x, y):
    """Closed-form 1D linear regression (y = m * x + c) to bypass MKL SVD issues."""
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xx = np.sum(x**2)
    sum_xy = np.sum(x * y)
    
    denom = n * sum_xx - sum_x**2
    if abs(denom) < 1e-9:
        return 0.0, float(np.mean(y))
        
    m = (n * sum_xy - sum_x * sum_y) / denom
    c = (sum_y - m * sum_x) / n
    return m, c


def fig_control_consigna_inversa(rows):
    # Parse registro_corridas.jsonl to get deflector voltages and corresponding beam offsets
    reg_pts = []
    log_path = ROOT / "studies" / "registro_corridas.jsonl"
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("fuente") == "simion" and d.get("features") is not None:
                        feats = d["features"]
                        volts = d["voltajes"]
                        ox = feats.get("offset_x_mm")
                        oy = feats.get("offset_y_mm")
                        v15 = volts.get("15") or volts.get("V15")
                        v18 = volts.get("18") or volts.get("V18")
                        hits = feats.get("hits") or d.get("hits", 0)
                        if all(val is not None for val in (ox, oy, v15, v18, hits)):
                            # Only include transmitted beam runs (hits > 10) and verify values are finite numbers
                            if float(hits) > 10 and np.isfinite(float(ox)) and np.isfinite(float(oy)):
                                reg_pts.append({
                                    "V15": float(v15),
                                    "V18": float(v18),
                                    "ox": float(ox),
                                    "oy": float(oy),
                                })
                except Exception:
                    pass
                    
    if len(reg_pts) < 5:
        # Fallback to sqlite rows metadata
        for r in rows:
            ox = r.get("offset_x")
            oy = r.get("offset_y")
            v15 = r["params"].get("V15")
            v18 = r["params"].get("V18")
            hits = r.get("hits") or 0
            if all(val is not None for val in (ox, oy, v15, v18)):
                if float(hits) > 10 and np.isfinite(float(ox)) and np.isfinite(float(oy)):
                    reg_pts.append({"V15": float(v15), "V18": float(v18), "ox": float(ox), "oy": float(oy)})
                    
    if len(reg_pts) < 3:
        print("--  sin datos para fig_control_consigna_inversa")
        return
        
    v15 = np.array([p["V15"] for p in reg_pts])
    v18 = np.array([p["V18"] for p in reg_pts])
    ox = np.array([p["ox"] for p in reg_pts])
    oy = np.array([p["oy"] for p in reg_pts])
    
    # Fits for Inverse Command (voltages needed for desired offsets)
    # V15 = mx * ox + cx (steering X)
    # V18 = my * oy + cy (steering Y)
    mx, cx = simple_linear_fit(ox, v15)
    my, cy = simple_linear_fit(oy, v18)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    
    # Panel A: Horizontal Deflection Calibration
    ax1.scatter(ox, v15, color="teal", edgecolor="k", lw=0.4, label="Datos SIMION")
    fit_x = np.linspace(ox.min() - 2, ox.max() + 2, 100)
    ax1.plot(fit_x, mx * fit_x + cx, "r-", label=rf"Ajuste Inverso: $V_{{15}} = {mx:.1f} \cdot x_{{off}} {cx:+.1f}$")
    ax1.set_xlabel("Desplazamiento horizontal en detector $x_{off}$ (mm)")
    ax1.set_ylabel("Voltaje Deflector Horizontal V15 (V)")
    ax1.set_title("Dirección Horizontal y Consigna Inversa")
    ax1.legend()
    ax1.grid(alpha=0.25)
    
    # Panel B: Vertical Deflection Calibration
    ax2.scatter(oy, v18, color="purple", edgecolor="k", lw=0.4, label="Datos SIMION")
    fit_y = np.linspace(oy.min() - 2, oy.max() + 2, 100)
    ax2.plot(fit_y, my * fit_y + cy, "r-", label=rf"Ajuste Inverso: $V_{{18}} = {my:.1f} \cdot y_{{off}} {cy:+.1f}$")
    ax2.set_xlabel("Desplazamiento vertical en detector $y_{off}$ (mm)")
    ax2.set_ylabel("Voltaje Deflector Vertical V18 (V)")
    ax2.set_title("Dirección Vertical y Consigna Inversa")
    ax2.legend()
    ax2.grid(alpha=0.25)
    
    fig.suptitle("Calibración del Control de Dirección y Consigna Inversa del Haz", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig_control_consigna_inversa.png", dpi=150)
    plt.close(fig)
    print("ok  fig_control_consigna_inversa.png")


def fig_convergencia_gemelo(rows):
    by_study = {}
    for r in rows:
        by_study.setdefault((r["label"], r["maximize"]), []).append(r)

    fig, ax_d = plt.subplots(figsize=(8, 6))
    for (label, maximize), rs in sorted(by_study.items()):
        # Filter for the relevant comparative studies
        if label not in ("gemelo_v2", "gemelo_db_v2"):
            continue
        rs = sorted(rs, key=lambda r: r["number"])
        vals = [r["value"] for r in rs]
        running = np.minimum.accumulate(vals)
        ax_d.plot(range(1, len(vals) + 1), running, marker=".", label=f"{label} (Costo $J_{{v2}}$)")
        
    ax_d.set_title("Convergencia: TPE (gemelo_v2) vs GP-Seeded (gemelo_db_v2)")
    ax_d.set_xlabel("Evaluación SIMION #")
    ax_d.set_ylabel("Mejor distancia / costo acumulado (mm, menor = mejor)")
    ax_d.grid(alpha=0.25)
    ax_d.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_convergencia_gemelo.png", dpi=150)
    plt.close(fig)
    print("ok  fig_convergencia_gemelo.png")


def fig_haz_detector():
    """UNA corrida SIMION real del mejor config, guardando los splats."""
    chosen = {int(k.lstrip("V")): float(v) for k, v in BEST_PARAMS.items()}
    print(f"Volando el mejor config en SIMION: { {k: round(v) for k, v in sorted(chosen.items())} }")
    op.apply_voltages(chosen)
    out = op.run_simion(op.FLY_COMMAND)
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
    fig_solucion_quad_3d(rows)
    fig_validacion_gemelo(rows)
    fig_incertidumbre_extrapolacion(rows)
    fig_control_consigna_inversa(rows)
    fig_convergencia_gemelo(rows)
    if "--sin-simion" not in sys.argv:
        fig_haz_detector()
    print(f"\nFiguras científicas guardadas en: {OUT}")
