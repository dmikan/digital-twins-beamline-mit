"""
perfil_envolvente.py
=====================

El "detector virtual cada 10mm": vuela el MEJOR config real (mas hits
SIMION archivados) y un no-hitter de contraste, y barre
caracterizador.cinematica_en_plano a lo largo de AMBOS tramos de la
linea. Muestra donde el haz pierde colimacion, donde estan las cinturas
(cambios de signo de alfa), y donde mueren las particulas -- el
diagnostico para decidir que electrodo retocar y donde exigir
colimacion.

Salida: outputs/perfil_envolvente.png + tabla de puntos criticos.

Correr:  python playpen/perfil_envolvente.py
"""

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

optuna.logging.set_verbosity(optuna.logging.WARNING)

from physics import IonSpecies
from physics import (
    BatchBasisFieldMap, make_batch_beam, BatchTrajectory, BatchRK4Integrator,
)
from caracterizador import make_beam, BeamProgressScorer
from physics import build_wall_index
from caracterizador import cinematica_en_plano
import optimizer as orch
orch.N_ELECTRODES = 19
from validate_rk4_filter import collect_archived_trials

N_PART = 200
DT = 1.0e-8
STEPS = 3000

# planos de barrido: tramo -x (fuente x=395 -> bender x~75) y
# tramo +z (bender z~77 -> detector z~405)
PLANOS_X = np.arange(385.0, 84.0, -10.0)   # eje 0, direccion -1
PLANOS_Z = np.arange(95.0, 401.0, 10.0)    # eje 2, direccion +1


def perfil(pos, vel_c, stop, pared, planos, eje, direccion):
    """Barrido de cinematica_en_plano -> dict de arrays alineados a planos."""
    campos = ("n_cruzan", "sigma_x_mm", "sigma_y_mm", "div_x_mrad", "div_y_mrad",
              "twiss_alpha_x", "twiss_alpha_y", "emittance_x", "emittance_y")
    out = {c: np.full(len(planos), np.nan) for c in campos}
    for i, plano in enumerate(planos):
        f = cinematica_en_plano(pos, vel_c, stop_idx=stop, paso_pared=pared,
                                z_plano=float(plano), eje=eje, direccion=direccion)
        for c in campos:
            out[c][i] = f[c]
    return out


def main():
    # v2 (2026-07-06): fisica PROMOVIDA (dual 1mm) via cargar_fisica, y
    # los configs salen del ESTUDIO gemelo_v2 (la familia record) -- la
    # recoleccion de DBs archivadas fluctua con OneDrive y ya nos vendio
    # un config "base" equivocado una vez. Contraste = mismo config con
    # el V6 viejo (-208): visualiza que hace el enfoque del tramo final.
    from gemelo import GemeloDigital
    from physics import cargar_fisica
    tw = GemeloDigital()
    m = tw.mejor()
    if m is None:
        raise SystemExit("Estudio vacio -- corre playpen/sembrar_record.py primero.")
    v_rec = tw.voltajes_completos(m["voltajes"])
    v_contraste = v_rec.copy()
    v_contraste[5] = -208.0
    print(f"Config: mejor del estudio (trial {m['trial']}, J={m['objetivo']:.3f}, "
          f"hits={m['hits']}) | contraste: mismo con V6=-208")

    fis = cargar_fisica(ROOT)
    bfm, wall_index = fis.bfm, fis.wall
    species = IonSpecies(mass=28 * 1.66053906660e-27, charge=1.602176634e-19)
    sp, sv = make_beam(N=N_PART, species=species, start_point=[395.0, 75.0, 77.0],
                       mean_energy_eV=15.0, std_energy_eV=0.42466, half_angle_deg=15.0, seed=42)

    par = np.stack([v_rec, v_contraste])
    bfm.set_voltages_batch(par)
    beam, ci = make_batch_beam(species, sp, sv, 2)
    traj = BatchTrajectory(beam, ci)
    print(f"Volando 2 configs x {N_PART} particulas x {STEPS} pasos...")
    BatchRK4Integrator(bfm, ci).integrate(traj, dt=DT, num_steps=STEPS)
    scorer = BeamProgressScorer(
        bfm=bfm, Trajectory=traj, dt=DT, num_steps=STEPS,
        detector_bbox=orch.DETECTOR_BBOX, wall_index=wall_index,
        wall_hit_margin=fis.margen,
        wall_check_midpoints=False, wall_check_stride=3)
    r = scorer.score(par)

    pos = r["positions"].reshape(-1, 2, N_PART, 3)
    stop = r["stop_idx"].reshape(2, N_PART)
    pared = r["hit_wall_step"].reshape(2, N_PART)

    perfiles = []
    for c in range(2):
        vel_c = np.asarray([s.velocity[c * N_PART:(c + 1) * N_PART] for s in traj.states],
                           dtype=np.float32)
        perfiles.append({
            "x": perfil(pos[:, c], vel_c, stop[c], pared[c], PLANOS_X, eje=0, direccion=-1),
            "z": perfil(pos[:, c], vel_c, stop[c], pared[c], PLANOS_Z, eje=2, direccion=+1),
        })

    # ------------------------------ figura ------------------------------
    filas = [("n_cruzan", "particulas limpias", None),
             ("sigma", "sigma (mm)", ("sigma_x_mm", "sigma_y_mm")),
             ("div", "divergencia (mrad)", ("div_x_mrad", "div_y_mrad")),
             ("alpha", "Twiss alfa", ("twiss_alpha_x", "twiss_alpha_y"))]
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex="col")
    nombres = [f"RECORD (V6={v_rec[5]:.0f})", "contraste (V6=-208)"]
    colores = ["tab:green", "tab:red"]

    for col, (tramo, planos, xlabel) in enumerate(
            (("x", PLANOS_X, "x (mm)  [fuente 395 -> bender ~75]"),
             ("z", PLANOS_Z, "z (mm)  [bender ~77 -> detector 405]"))):
        for fila, (clave, ylabel, subcl) in enumerate(filas):
            ax = axes[fila, col]
            for c in range(2):
                p = perfiles[c][tramo]
                if subcl is None:
                    ax.plot(planos, p["n_cruzan"], color=colores[c], label=nombres[c])
                else:
                    ax.plot(planos, p[subcl[0]], color=colores[c], ls="-",
                            label=f"{nombres[c]} (1er transversal)")
                    ax.plot(planos, p[subcl[1]], color=colores[c], ls="--", alpha=0.6,
                            label=f"{nombres[c]} (2do transversal)")
            if clave == "alpha":
                ax.axhline(0, color="k", lw=0.5)
                ax.set_ylim(-15, 15)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            if fila == 0:
                ax.set_title(f"tramo {tramo}")
                ax.legend(fontsize=7)
            if fila == 3:
                ax.set_xlabel(xlabel)
        if tramo == "x":
            for a in axes[:, col]:
                a.invert_xaxis()   # sentido del vuelo

    fig.suptitle("Perfil de envolvente RK4 -- mejor config vs no-hitter "
                 "(solo particulas que cruzan limpio cada plano)")
    fig.tight_layout()
    out_png = ROOT / "outputs" / "perfil_envolvente.png"
    fig.savefig(out_png, dpi=140)
    print(f"Figura: {out_png}")

    # -------------------------- puntos criticos --------------------------
    print("\n===== PUNTOS CRITICOS (mejor config) =====")
    for tramo, planos, eje_nom in (("x", PLANOS_X, "x"), ("z", PLANOS_Z, "z")):
        p = perfiles[0][tramo]
        n = p["n_cruzan"]
        caida = np.diff(n)
        if len(caida) and np.nanmin(caida) < 0:
            j = int(np.nanargmin(caida))
            print(f"  tramo {tramo}: mayor perdida de particulas entre {eje_nom}={planos[j]:g} "
                  f"y {eje_nom}={planos[j + 1]:g}  ({int(n[j])} -> {int(n[j + 1])})")
        for sufijo, nombre in (("_x", "1er transversal"), ("_y", "2do transversal")):
            a = p[f"twiss_alpha{sufijo}"]
            cruces = np.where(np.diff(np.sign(a[~np.isnan(a)])) != 0)[0]
            validos = planos[~np.isnan(a)]
            if len(cruces):
                donde = ", ".join(f"{eje_nom}={validos[i]:g}" for i in cruces[:4])
                print(f"  tramo {tramo}: cintura(s) {nombre} cerca de {donde}")
            d = p[f"div{sufijo}_mrad"]
            dd = np.abs(np.diff(d))
            if np.isfinite(dd).any():
                j = int(np.nanargmax(dd))
                print(f"  tramo {tramo}: mayor salto de divergencia {nombre} entre "
                      f"{eje_nom}={planos[j]:g} y {eje_nom}={planos[j + 1]:g} "
                      f"({d[j]:.0f} -> {d[j + 1]:.0f} mrad)")


if __name__ == "__main__":
    main()
