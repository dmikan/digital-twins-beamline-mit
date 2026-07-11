"""
scan_colimacion_prequad.py
===========================

MEJORAR LA COLIMACION ANTES DEL CUADRUPOLO (entrada del bender, x~120).

Diagnostico previo (perfil_envolvente.py): de 200 particulas, solo ~15
entran limpias al bender -- sigma_y ~6mm en x~135 contra un gap de 44mm
con optica de por medio. Los electrodos que enfocan ANTES del bender son
V3 (einzel 1) y V6 (einzel 2), ambos optimizables.

Fase 1 -- barrido 2D V3 x V6 (81 configs, solo RK4, sin SIMION):
    el resto de electrodos queda en el mejor config archivado. Se mide
    EN LA ENTRADA DEL CUADRUPOLO (x=135 y x=120, 200 particulas):
    n limpias, sigma, divergencia, puntaje cinematico.

Fase 2 -- linea completa para los top-5 de fase 1:
    vuelo largo (3000 pasos) y score base del screening + kin(z=390),
    para verificar que la colimacion pre-quad no arruina el tramo z.

Salidas: outputs/scan_colimacion_prequad.png (mapas V3xV6) + tablas.

Correr:  python playpen/scan_colimacion_prequad.py
"""

import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

optuna.logging.set_verbosity(optuna.logging.WARNING)

import optimizer as op
from physics import IonSpecies
from physics import (
    BatchBasisFieldMap, make_batch_beam, BatchTrajectory, BatchRK4Integrator,
)
from caracterizador import make_beam, BeamProgressScorer
from physics import build_wall_index
from caracterizador import cinematica_en_plano, puntaje_cinematico
import optimizer as orch
orch.N_ELECTRODES = 19
from validate_rk4_filter import collect_archived_trials

N_PART = 200
PASOS_CORTOS = 1200        # alcanza para cruzar el tramo -x
PASOS_LARGOS = 3000        # linea completa (fase 2)
DT = 1.0e-8
CHUNK = 10                 # 10 configs x 200 particulas: memoria comoda
V_GRID = np.linspace(-1000.0, 1000.0, 9)   # 9x9 = 81 combinaciones V3 x V6
X_QUAD = 120.0             # entrada del cuadrupolo
X_PRE = 135.0              # justo antes


def medir_prequad(pos, vel_c, stop, pared):
    """Metricas de colimacion en la entrada del cuadrupolo para UN config."""
    out = {}
    for etiq, plano in (("pre", X_PRE), ("quad", X_QUAD)):
        f = cinematica_en_plano(pos, vel_c, stop_idx=stop, paso_pared=pared,
                                z_plano=plano, eje=0, direccion=-1)
        out[f"n_{etiq}"] = f["n_cruzan"]
        out[f"sigma_y_{etiq}"] = f["sigma_x_mm"]     # 1er transversal de eje x = y
        out[f"sigma_z_{etiq}"] = f["sigma_y_mm"]     # 2do transversal = z
        out[f"kin_{etiq}"] = puntaje_cinematico(f)
    return out


def volar(bfm, wall_index, species, sp, sv, volts, pasos, medidor):
    """Vuela configs en chunks y aplica `medidor(pos, vel_c, stop, pared, extra)`
    por config; devuelve la lista de resultados. extra = (scorer_result, c, n_cfg)."""
    res = []
    for lo in range(0, len(volts), CHUNK):
        chunk = volts[lo:lo + CHUNK]
        n_cfg = chunk.shape[0]
        bfm.set_voltages_batch(chunk)
        beam, ci = make_batch_beam(species, sp, sv, n_cfg)
        traj = BatchTrajectory(beam, ci)
        BatchRK4Integrator(bfm, ci).integrate(traj, dt=DT, num_steps=pasos)
        scorer = BeamProgressScorer(
            bfm=bfm, Trajectory=traj, dt=DT, num_steps=pasos,
            detector_bbox=orch.DETECTOR_BBOX, wall_index=wall_index,
            wall_hit_margin=1.5, wall_check_midpoints=False, wall_check_stride=3)
        r = scorer.combined_score(chunk, **orch.SCORE_WEIGHTS)
        pos = r["positions"].reshape(-1, n_cfg, N_PART, 3)
        stop = r["stop_idx"].reshape(n_cfg, N_PART)
        pared = r["hit_wall_step"].reshape(n_cfg, N_PART)
        for c in range(n_cfg):
            vel_c = np.asarray([s.velocity[c * N_PART:(c + 1) * N_PART] for s in traj.states],
                               dtype=np.float32)
            res.append(medidor(pos[:, c], vel_c, stop[c], pared[c], (r, c, n_cfg)))
    return res


def main():
    voltages, hits = collect_archived_trials()
    i_best = int(np.argmax(hits))
    base_v = voltages[i_best].copy()
    print(f"Config base: mejor archivado ({hits[i_best]:g} hits) -- "
          f"V3={base_v[2]:.0f}, V6={base_v[5]:.0f} actuales")

    print("Cargando campo + paredes...")
    bfm = BatchBasisFieldMap.from_directory(ROOT, n_electrodes=orch.N_ELECTRODES)
    wall_index = build_wall_index(ROOT, exclude=orch.WALL_EXCLUDE, target_spacing=2.0)
    species = IonSpecies(mass=28 * 1.66053906660e-27, charge=1.602176634e-19)
    sp, sv = make_beam(N=N_PART, species=species, start_point=[395.0, 75.0, 77.0],
                       mean_energy_eV=15.0, std_energy_eV=0.42466, half_angle_deg=15.0, seed=42)

    # ---------------- fase 1: barrido V3 x V6, medir en el quad ----------------
    combos = [(v3, v6) for v3 in V_GRID for v6 in V_GRID]
    volts = np.tile(base_v, (len(combos), 1))
    for i, (v3, v6) in enumerate(combos):
        volts[i, 2], volts[i, 5] = v3, v6
    print(f"Fase 1: {len(combos)} combos V3xV6, {N_PART} particulas x {PASOS_CORTOS} pasos...")
    t0 = time.time()
    met = volar(bfm, wall_index, species, sp, sv, volts, PASOS_CORTOS,
                lambda pos, vel, stop, pared, extra: medir_prequad(pos, vel, stop, pared))
    print(f"  fase 1 en {time.time() - t0:.0f}s")

    n_quad = np.array([m["n_quad"] for m in met]).reshape(len(V_GRID), len(V_GRID))
    sig_pre = np.array([m["sigma_y_pre"] for m in met]).reshape(len(V_GRID), len(V_GRID))
    kin_pre = np.array([m["kin_pre"] for m in met]).reshape(len(V_GRID), len(V_GRID))

    # referencia: el config base actual
    j3 = int(np.argmin(np.abs(V_GRID - base_v[2])))
    j6 = int(np.argmin(np.abs(V_GRID - base_v[5])))
    print(f"\n  referencia (grid mas cercano al base): n_quad={n_quad[j3, j6]:.0f} limpias")

    orden = np.argsort([-m["n_quad"] + m["kin_pre"] for m in met])   # mas limpias, luego kin
    print(f"\n===== TOP 10 COLIMACION EN LA ENTRADA DEL CUADRUPOLO (x={X_QUAD:g}) =====")
    print(f"  {'V3':>7} {'V6':>7}  {'n_quad':>6}  {'sigma_y@135':>11}  {'kin@135':>7}")
    for i in orden[:10]:
        v3, v6 = combos[i]
        m = met[i]
        s = f"{m['sigma_y_pre']:.2f}" if np.isfinite(m["sigma_y_pre"]) else "  --"
        print(f"  {v3:>7.0f} {v6:>7.0f}  {m['n_quad']:>6.0f}  {s:>11}  {m['kin_pre']:>7.3f}")

    # ---------------- figura ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, dato, titulo in ((axes[0], n_quad, f"n limpias en x={X_QUAD:g} (entrada quad)"),
                             (axes[1], sig_pre, f"sigma_y (mm) en x={X_PRE:g}"),
                             (axes[2], kin_pre, f"puntaje cinematico en x={X_PRE:g}")):
        im = ax.imshow(dato, origin="lower", aspect="auto", cmap="viridis",
                       extent=[V_GRID[0], V_GRID[-1], V_GRID[0], V_GRID[-1]])
        ax.plot(base_v[5], base_v[2], "r*", ms=14, label="config actual")
        ax.set_xlabel("V6 (einzel 2)")
        ax.set_ylabel("V3 (einzel 1)")
        ax.set_title(titulo)
        ax.legend(loc="lower right", fontsize=8)
        fig.colorbar(im, ax=ax)
    fig.suptitle("Barrido V3 x V6: colimacion antes del cuadrupolo (resto = mejor config)")
    fig.tight_layout()
    out_png = ROOT / "outputs" / "scan_colimacion_prequad.png"
    fig.savefig(out_png, dpi=140)
    print(f"\nFigura: {out_png}")

    # ---------------- fase 2: linea completa para el top-5 ----------------
    top5 = [i for i in orden[:5]]
    volts5 = np.vstack([volts[i] for i in top5] + [base_v])   # + base como control
    nombres = [f"V3={combos[i][0]:.0f},V6={combos[i][1]:.0f}" for i in top5] + ["BASE actual"]
    print(f"\nFase 2: linea completa ({PASOS_LARGOS} pasos) para top-5 + base...")

    n_keep = max(1, int(np.ceil(N_PART * orch.SPLAT_TOP_FRACTION)))

    def medidor_full(pos, vel_c, stop, pared, extra):
        r, c, n_cfg = extra
        tdist = r["target_distance"].reshape(n_cfg, N_PART)[c]
        reached = r["reached_target"].reshape(n_cfg, N_PART)[c].mean()
        wall = r["hit_wall"].reshape(n_cfg, N_PART)[c].mean()
        score = (np.sort(tdist)[:n_keep].mean()
                 + orch.HIT_TERM_SCALE_MM * (1.0 - reached)
                 + orch.WALL_PENALTY_MM * wall)
        f390 = cinematica_en_plano(pos, vel_c, stop_idx=stop, paso_pared=pared,
                                   z_plano=390.0, eje=2, direccion=+1)
        m = medir_prequad(pos, vel_c, stop, pared)
        return dict(score_mm=float(score), reached=float(reached),
                    n_390=f390["n_cruzan"], kin_390=puntaje_cinematico(f390), **m)

    t0 = time.time()
    res5 = volar(bfm, wall_index, species, sp, sv, volts5, PASOS_LARGOS, medidor_full)
    print(f"  fase 2 en {time.time() - t0:.0f}s")

    print(f"\n===== LINEA COMPLETA: los colimados pre-quad, llegan mejor? =====")
    print(f"  {'config':<22} {'n_quad':>6} {'n_z390':>6} {'kin390':>7} {'score RK4 (mm)':>14}")
    for nombre, m in zip(nombres, res5):
        print(f"  {nombre:<22} {m['n_quad']:>6.0f} {m['n_390']:>6.0f} "
              f"{m['kin_390']:>7.3f} {m['score_mm']:>14.1f}")
    print("\n  score RK4: menor = mejor (el predictor validado del objetivo SIMION).")
    print("  Si un candidato domina en n_quad Y en score, es candidato directo a")
    print("  gemelo.evaluar() (vuelo SIMION real).")


if __name__ == "__main__":
    main()
