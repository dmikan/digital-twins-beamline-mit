"""
plot_rk4_trajectories.py
========================

Simula las trayectorias de los iones utilizando el integrador numérico RK4
del gemelo digital, y grafica:
  (a) Trayectorias en 3D (fig_rk4_trayectorias_3d.png).
  (b) Evolución del Espacio de Fases (fig_rk4_espacio_fases.png).
"""

import pathlib
import sys
import numpy as np
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gemelo import GemeloDigital
import physics as phys
import caracterizador as carac
import optimizer as op

OUT = ROOT / "outputs" / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)


def simulate_best_config():
    # 1. Load best voltages from gemelo_db_v2.db
    db_path = ROOT / "studies" / "gemelo_db_v2.db"
    try:
        study = optuna.load_study(study_name="gemelo_db_v2", storage=f"sqlite:///{db_path}")
        best_trial = min([t for t in study.trials if t.value is not None], key=lambda t: t.value)
        voltajes = best_trial.params
        print(f"Mejores voltajes cargados: {voltajes}")
    except Exception as e:
        # Fallback default voltages if study load fails
        print(f"Aviso: No se pudo cargar el estudio ({e}). Usando voltajes por defecto.")
        voltajes = {"V3": 295.7, "V6": 359.4, "V9": 116.8, "V10": 92.1, "V11": 145.2, "V12": -605.7, "V15": -314.8, "V18": -5.0}

    # 2. Setup digital twin facade and load physics
    gd = GemeloDigital()
    bfm, wall = gd.fisica
    v_arr = gd.voltajes_completos(voltajes)
    
    # 3. Create ion beam
    particulas = 150
    pasos = op.RESCREEN_STEPS
    dt = op.RESCREEN_DT
    
    pos0, vel0 = carac.make_beam(
        N=particulas, species=gd._species,
        start_point=[395.0, 75.0, 77.0], mean_energy_eV=15.0,
        std_energy_eV=0.42466, half_angle_deg=15.0, seed=1234
    )
    
    # 4. Integrate trajectory
    bfm.set_voltages_batch(v_arr[None, :])
    beam, ci = phys.make_batch_beam(gd._species, pos0, vel0, 1)
    tray = phys.BatchTrajectory(beam, ci)
    phys.BatchRK4Integrator(bfm, ci).integrate(tray, dt=dt, num_steps=pasos)
    
    # 5. Extract trajectory positions and velocities
    # Shape: (steps, particles, 3)
    pos_history = np.array([state.position for state in tray.states])
    vel_history = np.array([state.velocity for state in tray.states])
    
    # Filter out particles that hit the wall early to keep plot clean
    # scorer helps check wall hits
    scorer = carac.BeamProgressScorer(
        bfm=bfm, Trajectory=tray, dt=dt, num_steps=pasos,
        detector_bbox=op.DETECTOR_BBOX, wall_index=wall, wall_hit_margin=gd._margen,
        wall_check_midpoints=False, wall_check_stride=3,
    )
    r = scorer.combined_score(v_arr[None, :], **op.SCORE_WEIGHTS)
    limpias = ~r["hit_wall"] & ~r["lost"]
    
    return pos_history, vel_history, limpias, pasos, voltajes


def plot_3d_trajectories(pos_history, limpias, voltajes):
    # Setup digital twin facade and load physics data
    gd = GemeloDigital()
    bfm, wall = gd.fisica
    v_arr = gd.voltajes_completos(voltajes)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Plot ParedesPA geometry points in light gray background
    # We swap axes: matplotlib's X=X, Y=Z, Z=Y (floor is XZ, Y is vertical)
    wall_pts = wall.puntos
    corridor = (wall_pts[:, 0] >= 10) & (wall_pts[:, 0] <= 410) & \
               (wall_pts[:, 2] >= 30) & (wall_pts[:, 2] <= 660) & \
               (wall_pts[:, 1] >= 35) & (wall_pts[:, 1] <= 120)
    wall_filtered = wall_pts[corridor]
    
    if len(wall_filtered) > 0:
        # Downsample points for performance
        stride = max(1, len(wall_filtered) // 8000)
        ax.scatter(wall_filtered[::stride, 0], 
                   wall_filtered[::stride, 2], 
                   wall_filtered[::stride, 1], 
                   s=1.2, c="0.75", alpha=0.06, label="Electrodos (PA)", zorder=1)
    
    # 2. Plot 2D potential field contour slice at the Y-midplane (Y = 75mm)
    y_mid = 75.0
    y_idx = np.argmin(np.abs(bfm.y - y_mid))
    V_slice = np.tensordot(v_arr, bfm.basis_V[:, :, y_idx, :], axes=(0, 0))
    
    X_grid, Z_grid = np.meshgrid(bfm.x, bfm.z, indexing="ij")
    
    # Contourf on XZ plane, projected at vertical height Z=75.0 (which is physical Y=75.0)
    cs = ax.contourf(X_grid, Z_grid, V_slice, zdir='z', offset=y_mid, 
                     cmap="RdBu_r", alpha=0.18, levels=30, zorder=2)
    
    fig.colorbar(cs, ax=ax, label="Potencial Electrostático V(X, Y=75, Z) (V)", shrink=0.5, aspect=15)
    
    # 3. Plot trajectory lines
    num_to_plot = 18
    plotted = 0
    steps, N, _ = pos_history.shape
    
    for i in range(N):
        if not limpias[i]:
            continue
        
        x = pos_history[:, i, 0]
        z = pos_history[:, i, 2] # physical Z -> matplotlib Y
        y = pos_history[:, i, 1] # physical Y -> matplotlib Z
        
        ax.plot(x, z, y, color="tab:blue", alpha=0.45, lw=1.2, zorder=3)
        plotted += 1
        if plotted >= num_to_plot:
            break
            
    # Set labels (swapped: base plane is XZ, Y is vertical)
    ax.set_xlabel("X (mm)", fontweight='bold')
    ax.set_ylabel("Z (mm)", fontweight='bold')
    ax.set_zlabel("Y (mm)", fontweight='bold')
    
    # Set physical boundaries of the active workspace
    ax.set_xlim(35, 410)
    ax.set_ylim(40, 660) # matplotlib Y = physical Z
    ax.set_zlim(40, 110) # matplotlib Z = physical Y (vertical)
    
    ax.set_title("Trayectorias RK4 en el Plano de Flexión XZ\n"
                 "Con contornos de potencial en Y=75mm y geometría del colimador", fontsize=11, pad=15)
    ax.legend(loc="upper left")
    
    # Adjust standard viewing angle
    ax.view_init(elev=26, azim=-55)
    
    fig.tight_layout()
    fig.savefig(OUT / "fig_rk4_trayectorias_3d.png", dpi=150)
    plt.close(fig)
    print("ok  fig_rk4_trayectorias_3d.png")


def plot_phase_space_evolution(pos_history, vel_history, limpias, pasos):
    # Select three steps: 0 (Start), middle, and end
    step_start = 0
    step_mid = pasos // 2
    step_end = pasos - 1
    
    steps_to_plot = [step_start, step_mid, step_end]
    step_labels = ["Inicio (Paso 0)", f"Mitad (Paso {step_mid})", f"Final (Paso {step_end})"]
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # Units: mm and mm/us (equivalent to km/s)
    # Velocity conversion factor to mm/us
    v_factor = 1e-3
    
    for idx, step in enumerate(steps_to_plot):
        # Coordinates at this step
        x = pos_history[step, :, 0]
        y = pos_history[step, :, 1]
        vx = vel_history[step, :, 0] * v_factor
        vy = vel_history[step, :, 1] * v_factor
        
        # Center offsets for better emittance plotting
        x_c = x - np.mean(x[limpias])
        y_c = y - np.mean(y[limpias])
        vx_c = vx - np.mean(vx[limpias])
        vy_c = vy - np.mean(vy[limpias])
        
        # Top Row: Horizontal Phase Space (x vs vx)
        axes[0, idx].scatter(x_c[~limpias], vx_c[~limpias], color="gray", s=10, alpha=0.3, label="Chocó pared")
        axes[0, idx].scatter(x_c[limpias], vx_c[limpias], color="teal", s=18, alpha=0.8, edgecolor='k', lw=0.3, label="Transmitido")
        axes[0, idx].set_title(f"X - {step_labels[idx]}")
        axes[0, idx].set_xlabel("Posición relativa X (mm)")
        axes[0, idx].set_ylabel("Velocidad Vx (mm/µs)")
        axes[0, idx].grid(alpha=0.25)
        if idx == 0:
            axes[0, idx].legend()
            
        # Bottom Row: Vertical Phase Space (y vs vy)
        axes[1, idx].scatter(y_c[~limpias], vy_c[~limpias], color="gray", s=10, alpha=0.3)
        axes[1, idx].scatter(y_c[limpias], vy_c[limpias], color="purple", s=18, alpha=0.8, edgecolor='k', lw=0.3)
        axes[1, idx].set_title(f"Y - {step_labels[idx]}")
        axes[1, idx].set_xlabel("Posición relativa Y (mm)")
        axes[1, idx].set_ylabel("Velocidad Vy (mm/µs)")
        axes[1, idx].grid(alpha=0.25)
        
    fig.suptitle("Evolución del Espacio de Fases del Haz de Iones en el Gemelo Digital RK4", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rk4_espacio_fases.png", dpi=150)
    plt.close(fig)
    print("ok  fig_rk4_espacio_fases.png")


def plot_best_voltages(voltajes):
    labels = ["V3\n(Einzel 1)", "V6\n(Einzel 2)", "V9\n(Bender 1)", "V10\n(Bender 2)",
              "V11\n(Bender 3)", "V12\n(Bender 4)", "V15\n(Defl X)", "V18\n(Defl Y)"]
    keys = ["V3", "V6", "V9", "V10", "V11", "V12", "V15", "V18"]
    values = [voltajes[k] for k in keys]
    
    # Lenses: tab:blue, Benders: tab:green, Deflectors: tab:purple
    colors = ["tab:blue", "tab:blue", "tab:green", "tab:green", "tab:green", "tab:green", "tab:purple", "tab:purple"]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor='k', alpha=0.8, width=0.6)
    
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    
    # Label each bar with its exact voltage
    for bar in bars:
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        ax.annotate(f"{height:.1f}V",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points",
                    ha='center', va=va, fontsize=9, fontweight='bold')
                    
    ax.set_ylabel("Voltaje (V)")
    ax.set_title("Configuración Óptima de Voltajes del Sistema de Electrodos", fontsize=12, pad=15)
    ax.grid(axis='y', alpha=0.25)
    
    # Add margins to y-axis limits
    ymin, ymax = min(values), max(values)
    ax.set_ylim(ymin * 1.15 if ymin < 0 else -50, ymax * 1.15 if ymax > 0 else 50)
    
    fig.tight_layout()
    fig.savefig(OUT / "fig_mejores_voltajes.png", dpi=150)
    plt.close(fig)
    print("ok  fig_mejores_voltajes.png")


if __name__ == "__main__":
    pos_hist, vel_hist, limpias, pasos, voltajes = simulate_best_config()
    plot_3d_trajectories(pos_hist, limpias, voltajes)
    plot_phase_space_evolution(pos_hist, vel_hist, limpias, pasos)
    plot_best_voltages(voltajes)
    print(f"\nTodos los gráficos RK4 y voltajes guardados en: {OUT}")
