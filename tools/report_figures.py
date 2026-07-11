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
BEST_MODEL = None   # set in collect()


# Los estudios del reporte: TPE, GP-seeded, Bender 6D y Bender 7D.
STUDIES_REPORTE = ("gemelo_v2", "gemelo_db_v2", "gemelo_v2_bender6d", "gemelo_v2_bender7d")


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
    global BEST_PARAMS, BEST_MODEL
    rows = []
    best_hits = -1
    for label, study in all_studies():
        maximize = study.direction == optuna.study.StudyDirection.MAXIMIZE
        for t in study.trials:
            if t.value is None:
                continue
            
            is_reduced = any(k in t.params for k in ("A", "B", "C"))
            if is_reduced:
                required_keys = ("V3", "V6", "A", "B", "V15", "V18")
                if not all(k in t.params for k in required_keys):
                    continue
                # Map 6D/7D parameters back to physical electrode voltages (V3, V6, V9..V12, V15, V18)
                # so the rest of the plotting functions are fully compatible without modification.
                reduced = {}
                for k, v in t.params.items():
                    if k.startswith("V"):
                        try:
                            reduced[int(k[1:])] = v
                        except ValueError:
                            reduced[k] = v
                    else:
                        reduced[k] = v
                v9_12 = op.expand_bender(reduced["A"], reduced["B"], reduced.get("C", 0.0))
                flat_params = {
                    "V3": reduced.get(3, 0.0),
                    "V6": reduced.get(6, 0.0),
                    "V9": float(np.clip(v9_12[9], -1000.0, 1000.0)),
                    "V10": float(np.clip(v9_12[10], -1000.0, 1000.0)),
                    "V11": float(np.clip(v9_12[11], -1000.0, 1000.0)),
                    "V12": float(np.clip(v9_12[12], -1000.0, 1000.0)),
                    "V15": reduced.get(15, 0.0),
                    "V18": reduced.get(18, 0.0),
                }
            else:
                if not all(f"V{e}" in t.params for e in op.OPTIMIZE):
                    continue
                flat_params = t.params

            hits = t.user_attrs.get("simion_hits")
            if hits is None:
                hits = t.value if (maximize and t.value >= 0) else None
            
            rows.append(dict(
                label=label, number=t.number, params=flat_params, value=t.value,
                maximize=maximize, hits=hits,
                rk4=t.user_attrs.get("rk4_score"),
                offset_x=t.user_attrs.get("f_offset_x_mm"),
                offset_y=t.user_attrs.get("f_offset_y_mm"),
            ))
            if hits is not None and hits > best_hits:
                best_hits, BEST_PARAMS, BEST_MODEL = hits, flat_params, label
    print(f"{len(rows)} trials SIMION registrados; mejor: {best_hits:g} hits")
    return rows


def fig_solucion_quad_3d(rows):
    # Filter rows with hits > 0
    pts = [r for r in rows if r["hits"] is not None and r["hits"] > 0]
    if not pts:
        print("--  sin datos para fig_solucion_quad_3d")
        return
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = {
        "gemelo_v2": "tab:green",
        "gemelo_db_v2": "tab:purple",
        "gemelo_v2_bender6d": "tab:blue",
        "gemelo_v2_bender7d": "tab:orange"
    }
    labels = {
        "gemelo_v2": "TPE (gemelo_v2)",
        "gemelo_db_v2": "GP-Seeded (gemelo_db_v2)",
        "gemelo_v2_bender6d": "Bender 6D (gemelo_v2_bender6d)",
        "gemelo_v2_bender7d": "Bender 7D (gemelo_v2_bender7d)"
    }
    
    for model in STUDIES_REPORTE:
        m_pts = [r for r in pts if r["label"] == model]
        if not m_pts:
            continue
        v9 = np.array([r["params"]["V9"] for r in m_pts])
        v10 = np.array([r["params"]["V10"] for r in m_pts])
        v12 = np.array([r["params"]["V12"] for r in m_pts])
        hits = np.array([r["hits"] for r in m_pts])
        
        sizes = 15 + (hits / hits.max()) * 80 if len(hits) > 0 else 40
        ax.scatter(v9, v10, v12, s=sizes, c=colors[model], edgecolor='k', lw=0.4, label=labels[model], alpha=0.85)
        
        # Add 3D standard deviation volume (1-sigma ellipsoid)
        if len(m_pts) >= 3:
            v9_mean, v9_std = np.mean(v9), np.std(v9)
            v10_mean, v10_std = np.mean(v10), np.std(v10)
            v12_mean, v12_std = np.mean(v12), np.std(v12)
            
            # Clip minimum std to avoid flat sphere projection issues
            v9_std = max(v9_std, 15.0)
            v10_std = max(v10_std, 15.0)
            v12_std = max(v12_std, 15.0)
            
            # Sphere points
            u = np.linspace(0, 2 * np.pi, 20)
            v_ang = np.linspace(0, np.pi, 20)
            x_sphere = np.outer(np.cos(u), np.sin(v_ang))
            y_sphere = np.outer(np.sin(u), np.sin(v_ang))
            z_sphere = np.outer(np.ones(np.size(u)), np.cos(v_ang))
            
            # Scale and shift to create 1-sigma ellipsoid
            x_ell = v9_mean + v9_std * x_sphere
            y_ell = v10_mean + v10_std * y_sphere
            z_ell = v12_mean + v12_std * z_sphere
            
            ax.plot_wireframe(x_ell, y_ell, z_ell, color=colors[model], alpha=0.12, lw=0.6)
        
    ax.set_xlabel("Voltaje de electrodo V9 (V)")
    ax.set_ylabel("Voltaje de electrodo V10 (V)")
    ax.set_zlabel("Voltaje de electrodo V12 (V)")
    ax.set_title("Espacio de Soluciones del Cuadrupolo (3D) - Comparativa de Modelos\n"
                 "Muestra unicamente candidatos con transmision (hits > 0)")
    ax.legend(loc="upper left")
    ax.view_init(elev=25, azim=45)
    
    fig.tight_layout()
    fig.savefig(OUT / "fig_solucion_quad_3d.png", dpi=150)
    plt.close(fig)
    print("ok  fig_solucion_quad_3d.png")


def fig_solucion_bender_2d(rows):
    pts = [r for r in rows if r["hits"] is not None and r["hits"] > 0]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(8, 6.5))
    v9 = np.array([r["params"]["V9"] for r in pts])
    v12 = np.array([r["params"]["V12"] for r in pts])
    hits = np.array([r["hits"] for r in pts])
    
    # Plot the symmetric diagonal line V9 = V12 for reference
    diag = np.linspace(-1000, 1000, 100)
    ax.plot(diag, diag, 'r--', alpha=0.5, label="Subespacio Simetrico (V9 = V12)")
    
    sc = ax.scatter(v9, v12, c=hits, cmap="viridis", s=55, edgecolor='k', lw=0.4, zorder=3)
    plt.colorbar(sc, ax=ax, label="SIMION Hits")
    
    ax.set_xlabel("Voltaje de electrodo V9 (V)")
    ax.set_ylabel("Voltaje de electrodo V12 (V)")
    ax.set_title("Espacio de Soluciones del Bender: V9 vs V12\nMuestra la Cuenca Asimetrica y el Subespacio Simetrico")
    ax.legend()
    ax.grid(alpha=0.25)
    
    fig.tight_layout()
    fig.savefig(OUT / "fig_solucion_bender_2d.png", dpi=150)
    plt.close(fig)
    print("ok  fig_solucion_bender_2d.png")


def fig_solucion_lentes_2d(rows):
    pts = [r for r in rows if r["hits"] is not None and r["hits"] > 0]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(8, 6.5))
    v3 = np.array([r["params"]["V3"] for r in pts])
    v6 = np.array([r["params"]["V6"] for r in pts])
    hits = np.array([r["hits"] for r in pts])
    
    sc = ax.scatter(v3, v6, c=hits, cmap="viridis", s=55, edgecolor='k', lw=0.4, zorder=3)
    plt.colorbar(sc, ax=ax, label="SIMION Hits")
    
    ax.set_xlabel("Voltaje de Einzel Lens 1 V3 (V)")
    ax.set_ylabel("Voltaje de Einzel Lens 2 V6 (V)")
    ax.set_title("Espacio de Soluciones de Lentes Einzel: V3 vs V6\nCorrelacion de enfoque primario de la linea de haz")
    ax.grid(alpha=0.25)
    
    fig.tight_layout()
    fig.savefig(OUT / "fig_solucion_lentes_2d.png", dpi=150)
    plt.close(fig)
    print("ok  fig_solucion_lentes_2d.png")


def fig_validacion_gemelo(rows, model):
    # Only trials with localized RK4 scores for this model
    m_rows = [r for r in rows if r["label"] == model]
    pts = [r for r in m_rows if not r["maximize"] and r["rk4"] is not None]
    if not pts:
        print(f"--  sin datos para fig_validacion_gemelo en {model}")
        return
    
    # 1. Complete validation plot (all data, including J=2.0)
    x_all = np.array([r["rk4"] for r in pts])
    y_all = np.array([r["value"] for r in pts])
    hits_all = np.array([r["hits"] or 0 for r in pts])
    
    pr_all, _ = pearsonr(x_all, y_all) if len(x_all) > 2 else (0.0, 0.0)
    sr_all, _ = spearmanr(x_all, y_all) if len(x_all) > 2 else (0.0, 0.0)
    
    fig, ax = plt.subplots(figsize=(8, 6.5))
    miss_all = hits_all == 0
    ax.scatter(x_all[miss_all], y_all[miss_all], s=35, c="0.6", alpha=0.5, label="0 hits")
    if (~miss_all).any():
        sc = ax.scatter(x_all[~miss_all], y_all[~miss_all], s=75, c=hits_all[~miss_all], cmap="viridis",
                        edgecolor='k', lw=0.4, vmin=1, label="con hits", zorder=3)
        plt.colorbar(sc, ax=ax, label="SIMION Hits")
        
    ax.set_xlabel("Prediccion de costo del gemelo RK4 (score J en mm, espacial)")
    ax.set_ylabel("Costo real en SIMION (score J, adimensional)")
    ax.set_title(f"Validacion Completa del Gemelo (Todos los Datos, n={len(pts)}) - Modelo: {model}\n"
                 f"Pearson: {pr_all:+.3f}, Spearman: {sr_all:+.3f}", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.25)
    ax.set_xlim(0.0, max(x_all.max() + 5, 50.0))
    ax.set_ylim(-0.1, 2.2)
    fig.tight_layout()
    fig.savefig(OUT / model / "fig_validacion_gemelo_completa.png", dpi=150)
    plt.close(fig)
    
    # 2. Informative validation plot (showing only trials with transmission, hits > 0)
    pts_inf = [r for r in pts if r["hits"] is not None and r["hits"] > 0]
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
        
    ax.set_xlabel("Prediccion de costo del gemelo RK4 (score J en mm, espacial)")
    ax.set_ylabel("Costo real en SIMION (score J, adimensional)")
    ax.set_title(f"Validacion del Gemelo (Solo con Hits, n={len(pts_inf)}) - Modelo: {model}\n"
                 f"Pearson: {pr:+.3f}, Spearman: {sr:+.3f}", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.25)
    
    ax.set_xlim(x.min() - 2, x.max() + 2)
    ax.set_ylim(-0.05, 1.15)
    fig.tight_layout()
    fig.savefig(OUT / model / "fig_validacion_gemelo.png", dpi=150)
    plt.close(fig)
    print(f"ok  {model}/fig_validacion_gemelo.png + fig_validacion_gemelo_completa.png (n={len(pts_inf)})")


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


def generate_gp_sweep(X_train, y_train, hits, cols, var_name, model_best_params, model):
    col_idx = cols.index(int(var_name.lstrip("V")))
    
    x_plot_1d = np.linspace(-1000, 1000, 500)
    X_test = np.zeros((500, len(cols)))
    for i, e in enumerate(cols):
        X_test[:, i] = model_best_params[f"V{e}"]
    X_test[:, col_idx] = x_plot_1d
    
    # Set length-scale based on electrode sensitivity
    if var_name in ("V9", "V10", "V11", "V12"):
        ls = 100.0  # Bender electrodes (high sensitivity)
    elif var_name in ("V15", "V18"):
        ls = 150.0  # Deflectors (medium sensitivity)
    else:
        ls = 380.0  # Lenses (low sensitivity, smooth basin)
    
    y_pred, sigma = numpy_gp_predict_8d(X_train, y_train, X_test, length_scale=ls, sigma_f=0.5, noise=0.08)
    
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot(x_plot_1d, y_pred, 'b-', lw=1.8, label=rf"Corte de Prediccion GP ($\mu$ en optimo)")
    # Re-introduced standard deviation clouds (both 1-sigma and 2-sigma regions)
    ax.fill_between(x_plot_1d, y_pred - 1*sigma, y_pred + 1*sigma,
                    color='blue', alpha=0.25, label=r"Incertidumbre GP ($\pm 1\sigma$)")
    ax.fill_between(x_plot_1d, y_pred - 2*sigma, y_pred + 2*sigma,
                    color='blue', alpha=0.10, label=r"Incertidumbre GP ($\pm 2\sigma$)")
    
    # Observations projected on this specific dimension
    sc = ax.scatter(X_train[:, col_idx], y_train, c=hits, cmap="viridis", s=45,
                    edgecolor='k', lw=0.5, label="Observaciones (Proyeccion de nube 8D)", zorder=3)
    plt.colorbar(sc, ax=ax, label="SIMION Hits")
    
    ax.set_xlabel(f"Voltaje del electrodo {var_name} (V)")
    ax.set_ylabel("Costo de la funcion objetivo J (adimensional, menor = mejor)")
    ax.set_title(f"Incertidumbre y Extrapolacion del GP (Corte 1D de {var_name}) - Modelo: {model}\n"
                 f"La incertidumbre del GP se contrae cerca del valor recomendado ({var_name} ~ {model_best_params[var_name]:.1f} V)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    ax.set_ylim(-0.1, 1.25)
    
    fig.tight_layout()
    filename = f"fig_incertidumbre_{var_name}.png"
    fig.savefig(OUT / model / filename, dpi=150)
    plt.close(fig)
    print(f"ok  {model}/{filename}")
 
 
def fig_incertidumbre_extrapolacion(rows, model):
    # Only trials for this model
    m_rows = [r for r in rows if r["label"] == model]
    pts = [r for r in m_rows if r["hits"] is not None]
    if len(pts) < 5:
        print(f"--  insuficientes datos para fig_incertidumbre_extrapolacion en {model}")
        return
    
    # Feature matrix: shapes (N, 8) corresponding to op.OPTIMIZE
    cols = sorted(op.OPTIMIZE)
    X_train = np.array([[r["params"][f"V{e}"] for e in cols] for r in pts])
    y_train = np.array([r["value"] for r in pts])
    hits = np.array([r["hits"] for r in pts])
    
    # Find best params for this specific model
    best_pt = min(pts, key=lambda r: r["value"])
    model_best_params = best_pt["params"]
    
    # Generate sweeps for all 8 optimized electrodes (V3, V6, V9, V10, V11, V12, V15, V18)
    for e in (3, 6, 9, 10, 11, 12, 15, 18):
        generate_gp_sweep(X_train, y_train, hits, cols, f"V{e}", model_best_params, model)


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


def fig_control_consigna_inversa(rows, model):
    # Parse registro_corridas.jsonl to get deflector voltages and corresponding beam offsets
    reg_pts = []
    # Identify unique V15, V18 combinations evaluated for this model in the sqlite db
    model_configs = set()
    for r in rows:
        if r["label"] == model:
            v15 = r["params"].get("V15")
            v18 = r["params"].get("V18")
            if v15 is not None and v18 is not None:
                model_configs.add((round(float(v15), 1), round(float(v18), 1)))

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
                            if (round(float(v15), 1), round(float(v18), 1)) in model_configs:
                                if float(hits) > 10 and np.isfinite(float(ox)) and np.isfinite(float(oy)):
                                    # Filter outliers: check if beam actually landed near/on the detector surface
                                    if abs(float(ox)) <= 12.0 and abs(float(oy)) <= 12.0:
                                        reg_pts.append({
                                            "V15": float(v15),
                                            "V18": float(v18),
                                            "ox": float(ox),
                                            "oy": float(oy),
                                        })
                except Exception:
                    pass
                    
    if len(reg_pts) < 5:
        # Fallback to sqlite rows metadata for this model
        for r in rows:
            if r["label"] == model:
                ox = r.get("offset_x")
                oy = r.get("offset_y")
                v15 = r["params"].get("V15")
                v18 = r["params"].get("V18")
                hits = r.get("hits") or 0
                if all(val is not None for val in (ox, oy, v15, v18)):
                    if float(hits) > 10 and np.isfinite(float(ox)) and np.isfinite(float(oy)):
                        if abs(float(ox)) <= 12.0 and abs(float(oy)) <= 12.0:
                            reg_pts.append({"V15": float(v15), "V18": float(v18), "ox": float(ox), "oy": float(oy)})
                    
    if len(reg_pts) < 3:
        print(f"--  sin datos para fig_control_consigna_inversa en {model}")
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
    ax1.set_xlabel("Desplazamiento horizontal del centroide x_off (mm)")
    ax1.set_ylabel("Voltaje de placas deflectoras horizontales V15 (V)")
    ax1.set_title("Direccion Horizontal y Consigna Inversa")
    ax1.legend()
    ax1.grid(alpha=0.25)
    
    # Panel B: Vertical Deflection Calibration
    ax2.scatter(oy, v18, color="purple", edgecolor="k", lw=0.4, label="Datos SIMION")
    fit_y = np.linspace(oy.min() - 2, oy.max() + 2, 100)
    ax2.plot(fit_y, my * fit_y + cy, "r-", label=rf"Ajuste Inverso: $V_{{18}} = {my:.1f} \cdot y_{{off}} {cy:+.1f}$")
    ax2.set_xlabel("Desplazamiento vertical del centroide y_off (mm)")
    ax2.set_ylabel("Voltaje de placas deflectoras verticales V18 (V)")
    ax2.set_title("Direccion Vertical y Consigna Inversa")
    ax2.legend()
    ax2.grid(alpha=0.25)
    
    fig.suptitle(f"Calibracion del Control de Direccion y Consigna Inversa del Haz - Modelo: {model}", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / model / "fig_control_consigna_inversa.png", dpi=150)
    plt.close(fig)
    print(f"ok  {model}/fig_control_consigna_inversa.png")


def fig_convergencia_gemelo(rows):
    by_study = {}
    for r in rows:
        by_study.setdefault((r["label"], r["maximize"]), []).append(r)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    
    LABELS_MAP = {
        "gemelo_v2": "TPE (8D)",
        "gemelo_db_v2": "GP-Seeded (8D)",
        "gemelo_v2_bender6d": "Bender 6D",
        "gemelo_v2_bender7d": "Bender 7D"
    }
    COLORS_MAP = {
        "gemelo_v2": "tab:green",
        "gemelo_db_v2": "tab:purple",
        "gemelo_v2_bender6d": "tab:blue",
        "gemelo_v2_bender7d": "tab:orange"
    }

    for (label, maximize), rs in sorted(by_study.items()):
        if label not in STUDIES_REPORTE:
            continue
        rs = sorted(rs, key=lambda r: r["number"])
        vals = [r["value"] for r in rs]
        running = np.minimum.accumulate(vals)
        
        lbl = LABELS_MAP.get(label, label)
        col = COLORS_MAP.get(label, None)
        
        # Plot on both axes
        ax1.plot(range(1, len(vals) + 1), running, marker=".", color=col, label=lbl)
        ax2.plot(range(1, len(vals) + 1), running, marker=".", color=col, label=lbl)

    # 1. Left axis: General view
    ax1.set_title("Convergencia General (Toda la escala)")
    ax1.set_xlabel("Numero de evaluacion en SIMION")
    ax1.set_ylabel("Costo acumulado J_v2 (adimensional)")
    ax1.grid(alpha=0.25)
    ax1.set_ylim(0.50, 2.10)
    ax1.legend(loc="upper right")

    # 2. Right axis: Zoomed-in for discrimination
    ax2.set_title("Detalle de Convergencia (Zoom de Optimos)")
    ax2.set_xlabel("Numero de evaluacion en SIMION")
    ax2.set_ylabel("Costo acumulado J_v2 (Zoom)")
    ax2.grid(alpha=0.25)
    ax2.set_ylim(0.55, 0.70)  # Zoom in the region of the best costs (0.58 - 0.65)
    ax2.legend(loc="upper right")

    fig.suptitle(f"Comparativa de Convergencia de Optimizacion (Menor J_v2 = Mejor) | Mejor modelo: {LABELS_MAP.get(BEST_MODEL, BEST_MODEL)}", 
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_convergencia_gemelo.png", dpi=150)
    plt.close(fig)
    print("ok  fig_convergencia_gemelo.png (2 subplots: general + zoom)")


def fly_config(model_best_params, model, fig_name, csv_name, best_hitter=False):
    chosen = {int(k.lstrip("V")): float(v) for k, v in model_best_params.items()}
    print(f"[{model}] Volando {'best hitter' if best_hitter else 'best value'} en SIMION: { {k: round(v) for k, v in sorted(chosen.items())} }")
    op.apply_voltages(chosen)
    out = op.run_simion(op.FLY_COMMAND)
    positions = op.get_positions(out)
    hits = op.count_hits(positions)
    np.savetxt(OUT / model / csv_name, positions, delimiter=",",
               header="x_mm,y_mm,z_mm", comments="")
    print(f"    {positions.shape[0]} splats registrados, {hits} hits")

    xr, yr, zr = op.DETECTOR_REGION["x"], op.DETECTOR_REGION["y"], op.DETECTOR_REGION["z"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(positions[:, 2], bins=60, color="steelblue")
    ax1.axvspan(zr[0], zr[1], color="tab:green", alpha=0.4,
                label=f"detector z={zr[0]}-{zr[1]}")
    ax1.set_xlabel("Coordenada Z de colision/splat (mm)")
    ax1.set_ylabel("Numero de iones")
    ax1.set_title("Ubicacion Z final de impacto (500 iones)")
    ax1.legend()

    near = positions[positions[:, 2] > 350]
    ax2.scatter(near[:, 0], near[:, 1], s=18, c=near[:, 2], cmap="plasma")
    ax2.add_patch(plt.Rectangle((xr[0], yr[0]), xr[1] - xr[0], yr[1] - yr[0],
                                fill=False, edgecolor="tab:green", lw=2,
                                label="ventana del detector"))
    ax2.set_xlabel("Posicion X en detector (mm)")
    ax2.set_ylabel("Posicion Y en detector (mm)")
    ax2.set_title(f"Impactos transversales en z > 350mm (n={len(near)})")
    ax2.legend()
    ax2.set_aspect("equal")

    type_str = "Best Hitter" if best_hitter else "Best Value"
    fig.suptitle(
        f"Forma real del haz en Detector (SIMION) - Modelo: {model} ({type_str})\n"
        f"Voltajes recomendados (V): V3={model_best_params['V3']:.0f}, V6={model_best_params['V6']:.0f}, "
        f"V9={model_best_params['V9']:.0f}, V10={model_best_params['V10']:.0f}, V11={model_best_params['V11']:.0f}, "
        f"V12={model_best_params['V12']:.0f}, V15={model_best_params['V15']:.0f}, V18={model_best_params['V18']:.0f} | Hits: {hits}/500",
        fontsize=11
    )
    fig.tight_layout()
    fig.savefig(OUT / model / fig_name, dpi=150)
    plt.close(fig)
    print(f"ok  {model}/{fig_name} + {csv_name}")


def fig_haz_detector(model, rows):
    """UNA corrida SIMION real del mejor config de este modelo, guardando los splats."""
    model_rows = [r for r in rows if r["label"] == model]
    if not model_rows:
        print(f"--  sin datos para fig_haz_detector en {model}")
        return
    best_t = min(model_rows, key=lambda r: r["value"])
    
    # 1. Fly the best value config
    fly_config(best_t["params"], model, "fig_haz_en_detector.png", "splats_mejor_config.csv", best_hitter=False)
    
    # 2. Check if best hitter is different from best value
    best_hitter_t = max(model_rows, key=lambda r: r["hits"] if r["hits"] is not None else -1)
    if best_hitter_t["hits"] is not None and best_hitter_t["number"] != best_t["number"]:
        print(f"[{model}] Generando fig_haz_en_detector para best hitter (Trial #{best_hitter_t['number']})")
        fly_config(best_hitter_t["params"], model, "fig_haz_en_detector_best_hitter.png", "splats_mejor_config_best_hitter.csv", best_hitter=True)


def fig_cobertura_espacio(rows):
    # Plot V9 vs V12 for all trials of the three studies
    fig, ax = plt.subplots(figsize=(8, 6.5))
    
    colors_miss = {
        "gemelo_v2": "#A3E4D7",       # Light green
        "gemelo_db_v2": "#D7BDE2",    # Light purple
        "gemelo_v2_bender6d": "#AED6F1", # Light blue
        "gemelo_v2_bender7d": "#FDEBD0" # Light orange
    }
    colors_hit = {
        "gemelo_v2": "tab:green",
        "gemelo_db_v2": "tab:purple",
        "gemelo_v2_bender6d": "tab:blue",
        "gemelo_v2_bender7d": "tab:orange"
    }
    labels = {
        "gemelo_v2": "TPE (gemelo_v2)",
        "gemelo_db_v2": "GP-Seeded (gemelo_db_v2)",
        "gemelo_v2_bender6d": "Bender 6D (gemelo_v2_bender6d)",
        "gemelo_v2_bender7d": "Bender 7D (gemelo_v2_bender7d)"
    }
    
    # We plot the diagonal V9 = V12 representing the 6D quad subspace
    ax.plot([-1000, 1000], [-1000, 1000], 'k--', alpha=0.3, label="Subespacio Cuadrupolar (V9 = V12)")
    
    for model in STUDIES_REPORTE:
        m_rows = [r for r in rows if r["label"] == model]
        if not m_rows:
            continue
        v9 = np.array([r["params"]["V9"] for r in m_rows])
        v12 = np.array([r["params"]["V12"] for r in m_rows])
        hits = np.array([r["hits"] or 0 for r in m_rows])
        
        miss = hits == 0
        # Plot miss points
        ax.scatter(v9[miss], v12[miss], s=15, c=colors_miss[model], alpha=0.3, zorder=2)
        # Plot hit points
        if (~miss).any():
            ax.scatter(v9[~miss], v12[~miss], s=55, c=colors_hit[model], edgecolor='k', lw=0.4, 
                       label=f"{labels[model]} (con hits)", zorder=3)
            
    ax.set_xlabel("Voltaje de electrodo V9 (V)")
    ax.set_ylabel("Voltaje de electrodo V12 (V)")
    ax.set_title("Cobertura de Espacio de Busqueda: Voltaje V9 vs V12\n"
                 "Bender 6D se confina exactamente a la diagonal cuadrupolar (V9 = V12)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    ax.set_xlim(-1100, 1100)
    ax.set_ylim(-1100, 1100)
    
    fig.tight_layout()
    fig.savefig(OUT / "fig_cobertura_espacio.png", dpi=150)
    plt.close(fig)
    print("ok  fig_cobertura_espacio.png")


if __name__ == "__main__":
    rows = collect()
    
    # 1. Comparative/global plots (saved directly in OUT)
    fig_solucion_quad_3d(rows)
    fig_solucion_bender_2d(rows)
    fig_solucion_lentes_2d(rows)
    fig_convergencia_gemelo(rows)
    fig_cobertura_espacio(rows)
    
    # 2. Model-specific plots (saved in OUT / model /)
    for model in STUDIES_REPORTE:
        print(f"\nGenerando analisis para el gemelo: {model}")
        (OUT / model).mkdir(parents=True, exist_ok=True)
        
        fig_validacion_gemelo(rows, model)
        fig_incertidumbre_extrapolacion(rows, model)
        fig_control_consigna_inversa(rows, model)
        if "--sin-simion" not in sys.argv:
            fig_haz_detector(model, rows)
            
    print(f"\nFiguras cientificas guardadas en: {OUT}")
