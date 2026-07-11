"""Verificacion: puede el RK4 usar EL MISMO target que SIMION (J_v2 con
posicion + velocidad + pared) si le damos suficientes particulas para que
las "limpias" (sin choque ni perdida) soporten las features de forma?

Prueba directa contra verdad de terreno: los 20 trials del estudio
gemelo_v2 tienen su J_v2 real medido en SIMION. Se re-vuelan los 20
configs en RK4 a dos poblaciones (200 y 1000 particulas) y se comparan
tres scores RK4 contra el J real:
  mm     score de screening validado (acercamiento + reach + pared)
  Jv2    J_v2 con forma sobre TODAS las particulas (contaminado)
  Jv2L   J_v2 con forma sobre las LIMPIAS (mascara ~hit_wall & ~lost)

Salida: playpen/verificacion_target_rk4.txt
"""
import gc
import pathlib
import sys
import time

import numpy as np
import optuna
from scipy.stats import pearsonr, spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
optuna.logging.set_verbosity(optuna.logging.WARNING)

import optimizer as op
import orchestrator as orc
from RK4_sim_basis import IonSpecies
from RK4_sim_basis_batch import (
    BatchBasisFieldMap, make_batch_beam, BatchTrajectory, BatchRK4Integrator,
)
from beam_progress_score import make_beam, BeamProgressScorer
from caracterizador import caracterizar, objetivo_v2
from electrode_geometry import build_wall_index

L = []
def w(s=""):
    print(s)
    L.append(str(s))

# ---------------------------------------------------------------- trials
st = optuna.load_study(study_name="gemelo_v2",
                       storage=f"sqlite:///{ROOT / 'studies' / 'gemelo_v2.db'}")
trials = sorted([t for t in st.trials if t.value is not None], key=lambda t: t.number)
J_real = np.array([t.value for t in trials])
hits_real = np.array([t.user_attrs.get("simion_hits", 0) for t in trials])
volts = np.zeros((len(trials), orc.N_ELECTRODES))
for i, t in enumerate(trials):
    for e, vf in op.FIXED.items():
        volts[i, e - 1] = vf
    for e in op.OPTIMIZE:
        volts[i, e - 1] = t.params[f"V{e}"]
w(f"{len(trials)} configs con J_v2 real (hitters: {(hits_real > 0).sum()})")

print("cargando fisica...")
bfm = BatchBasisFieldMap.from_directory(ROOT, n_electrodes=orc.N_ELECTRODES)
wall = build_wall_index(ROOT, exclude=orc.WALL_EXCLUDE, target_spacing=2.0)
species = IonSpecies(mass=28 * 1.66053906660e-27, charge=1.602176634e-19)


def evaluar_rk4(n_particulas, chunk):
    pos0, vel0 = make_beam(N=n_particulas, species=species,
                           start_point=[395.0, 75.0, 77.0], mean_energy_eV=15.0,
                           std_energy_eV=0.42466, half_angle_deg=15.0, seed=1234)
    M = len(volts)
    mm = np.empty(M)
    jv2 = np.empty(M)
    jv2L = np.empty(M)
    n_limp = np.empty(M, dtype=int)
    n_keep = max(1, int(np.ceil(n_particulas * orc.SPLAT_TOP_FRACTION)))

    for lo in range(0, M, chunk):
        hi = min(lo + chunk, M)
        v = volts[lo:hi]
        ncfg = v.shape[0]
        bfm.set_voltages_batch(v)
        beam, ci = make_batch_beam(species, pos0, vel0, ncfg)
        tray = BatchTrajectory(beam, ci)
        BatchRK4Integrator(bfm, ci).integrate(tray, dt=orc.RESCREEN_DT,
                                              num_steps=orc.RESCREEN_STEPS)
        scorer = BeamProgressScorer(
            bfm=bfm, Trajectory=tray, dt=orc.RESCREEN_DT, num_steps=orc.RESCREEN_STEPS,
            detector_bbox=orc.DETECTOR_BBOX, wall_index=wall, wall_hit_margin=1.5,
            wall_check_midpoints=False, wall_check_stride=3)
        r = scorer.combined_score(v, **orc.SCORE_WEIGHTS)

        posiciones = r["positions"]
        stop = r["stop_idx"]
        idx = np.arange(posiciones.shape[1])
        vels = np.array([s.velocity for s in tray.states])
        fp = posiciones[stop, idx].reshape(ncfg, n_particulas, 3)
        fv = vels[stop, idx].reshape(ncfg, n_particulas, 3)
        td = r["target_distance"].reshape(ncfg, n_particulas)
        re_ = r["reached_target"].reshape(ncfg, n_particulas)
        lo_ = r["lost"].reshape(ncfg, n_particulas)
        wa = r["hit_wall"].reshape(ncfg, n_particulas)

        for c in range(ncfg):
            g = lo + c
            punta = float(np.sort(td[c])[:n_keep].mean())
            mm[g] = (punta + orc.HIT_TERM_SCALE_MM * (1 - re_[c].mean())
                     + orc.WALL_PENALTY_MM * wa[c].mean())
            flags = dict(reached=re_[c], lost=lo_[c], hit_wall=wa[c])
            limpias = ~wa[c] & ~lo_[c]
            n_limp[g] = int(limpias.sum())

            f_all = caracterizar(fp[c], fv[c], flags=flags)
            f_all["dist_punta_mm"] = punta
            jv2[g], _ = objetivo_v2(f_all)

            f_L = caracterizar(fp[c], fv[c],
                               mascara=limpias if limpias.sum() >= 3 else None,
                               flags=flags)
            f_L["dist_punta_mm"] = punta
            jv2L[g], _ = objetivo_v2(f_L)
        # liberar los arreglos grandes del chunk antes del siguiente
        del posiciones, vels, tray, beam, r, fp, fv
        gc.collect()
    return mm, jv2, jv2L, n_limp


for n_part, chunk in ((200, 20), (1000, 2)):
    t0 = time.time()
    mm, jv2, jv2L, n_limp = evaluar_rk4(n_part, chunk)
    w(f"\n===== N = {n_part} particulas ({time.time()-t0:.0f}s) =====")
    w(f"limpias por config: mediana {int(np.median(n_limp))}, "
      f"min {n_limp.min()}, max {n_limp.max()}")
    for nombre, s in (("mm (validado)  ", mm), ("Jv2 (todas)    ", jv2),
                      ("Jv2 (limpias)  ", jv2L)):
        pr, _ = pearsonr(s, J_real)
        sr, _ = spearmanr(s, J_real)
        # el hitter real es el J minimo: en que puesto lo rankea cada score?
        rank_hitter = int(np.argsort(s).tolist().index(int(np.argmin(J_real)))) + 1
        w(f"  {nombre}: Pearson {pr:+.3f}  Spearman {sr:+.3f}  "
          f"hitter real rankeado #{rank_hitter}/20")

(pathlib.Path(__file__).parent / "verificacion_target_rk4.txt").write_text(
    "\n".join(L), encoding="utf-8")
print("\nreporte: playpen/verificacion_target_rk4.txt")
