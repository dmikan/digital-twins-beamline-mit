"""
buscar_planos_criticos.py
==========================

QUE EL RK4 RECONOZCA DONDE HAY QUE GARANTIZAR COLIMACION.

En vez de decretar los planos (la leccion de calibrar_planos_kin.py:
promediarlos a ciegas diluye), barre ~19 planos candidatos por TODA la
linea sobre los 150 configs archivados con hits SIMION conocidos y mide,
por plano:

  1. separacion: puntaje cinematico medio de hitters vs todos (un plano
     "critico" es donde los futuros hitters YA se distinguen)
  2. ranking solo:  base + 20*kin(plano)
  3. ranking combo: base + 20*promedio(kin(plano), kin(z=390)) -- si un
     plano intermedio (p.ej. la entrada del cuadrupolo) APORTA senal
     nueva sobre la que z=390 ya captura, el combo supera 8/14

El plano intermedio que gane en combo es el "momento donde garantizar
colimacion" con respaldo de datos -- candidato a entrar en KIN_PLANES.

Correr:  python playpen/buscar_planos_criticos.py
"""

import pathlib
import sys
import time

import numpy as np
import optuna
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

optuna.logging.set_verbosity(optuna.logging.WARNING)

from RK4_sim_basis import IonSpecies
from RK4_sim_basis_batch import (
    BatchBasisFieldMap, make_batch_beam, BatchTrajectory, BatchRK4Integrator,
)
from beam_progress_score import make_beam, BeamProgressScorer
from electrode_geometry import build_wall_index
from caracterizador import cinematica_en_plano, puntaje_cinematico
import orchestrator as orch
from validate_rk4_filter import collect_archived_trials

SCREEN_N, SCREEN_STEPS, CHUNK, W = 50, 1500, 50, 20.0

# candidatos: toda la linea, ambos tramos (el ultimo es el z=390 actual)
CANDIDATOS = ([(0, -1, x) for x in (360.0, 330.0, 300.0, 270.0, 240.0, 210.0,
                                    180.0, 150.0, 120.0)] +
              [(2, +1, z) for z in (120.0, 150.0, 180.0, 210.0, 240.0, 270.0,
                                    300.0, 330.0, 360.0, 390.0)])
ETIQ = [f"{'x' if e == 0 else 'z'}={c:g}" for e, d, c in CANDIDATOS]
J390 = len(CANDIDATOS) - 1   # indice del plano de referencia z=390


def main():
    voltages, hits = collect_archived_trials()
    print("Cargando campo + paredes...")
    bfm = BatchBasisFieldMap.from_directory(ROOT, n_electrodes=orch.N_ELECTRODES)
    wall_index = build_wall_index(ROOT, exclude=orch.WALL_EXCLUDE, target_spacing=2.0)
    species = IonSpecies(mass=28 * 1.66053906660e-27, charge=1.602176634e-19)
    sp, sv = make_beam(N=SCREEN_N, species=species, start_point=[395.0, 75.0, 77.0],
                       mean_energy_eV=15.0, std_energy_eV=0.42466, half_angle_deg=15.0, seed=42)

    M, n_pl = len(voltages), len(CANDIDATOS)
    base = np.empty(M)
    kin = np.empty((n_pl, M))
    n_keep = max(1, int(np.ceil(SCREEN_N * orch.SPLAT_TOP_FRACTION)))

    print(f"Volando {M} sets, midiendo {n_pl} planos candidatos por config...")
    t0 = time.time()
    for lo in range(0, M, CHUNK):
        hi = min(lo + CHUNK, M)
        chunk = voltages[lo:hi]
        n_cfg = chunk.shape[0]

        bfm.set_voltages_batch(chunk)
        beam, ci = make_batch_beam(species, sp, sv, n_cfg)
        traj = BatchTrajectory(beam, ci)
        BatchRK4Integrator(bfm, ci).integrate(traj, dt=orch.DT, num_steps=SCREEN_STEPS)
        scorer = BeamProgressScorer(
            bfm=bfm, Trajectory=traj, dt=orch.DT, num_steps=SCREEN_STEPS,
            detector_bbox=orch.DETECTOR_BBOX, wall_index=wall_index, wall_hit_margin=1.5,
            wall_check_midpoints=False, wall_check_stride=3)
        r = scorer.combined_score(chunk, **orch.SCORE_WEIGHTS)

        tdist = r["target_distance"].reshape(n_cfg, SCREEN_N)
        reached = r["reached_target"].reshape(n_cfg, SCREEN_N).mean(axis=1)
        wall = r["hit_wall"].reshape(n_cfg, SCREEN_N).mean(axis=1)
        base[lo:hi] = (np.sort(tdist, axis=1)[:, :n_keep].mean(axis=1)
                       + orch.HIT_TERM_SCALE_MM * (1.0 - reached)
                       + orch.WALL_PENALTY_MM * wall)

        pos = r["positions"].reshape(-1, n_cfg, SCREEN_N, 3)
        stop_r = r["stop_idx"].reshape(n_cfg, SCREEN_N)
        pared_r = r["hit_wall_step"].reshape(n_cfg, SCREEN_N)
        for c in range(n_cfg):
            sl = slice(c * SCREEN_N, (c + 1) * SCREEN_N)
            vel_c = np.asarray([s.velocity[sl] for s in traj.states], dtype=np.float32)
            for j, (eje, d, coord) in enumerate(CANDIDATOS):
                f = cinematica_en_plano(pos[:, c], vel_c, stop_idx=stop_r[c],
                                        paso_pared=pared_r[c], z_plano=coord,
                                        eje=eje, direccion=d)
                kin[j, lo + c] = puntaje_cinematico(f)
        print(f"  configs {lo}-{hi - 1}  ({time.time() - t0:.0f}s)")

    hitter_idx = set(np.where(hits > 0)[0])
    hit_mask = hits > 0

    def metricas(kin_v):
        s = base + W * kin_v
        sr, _ = spearmanr(-s, hits)
        top30 = set(int(i) for i in np.argsort(s)[:30])
        return sr, len(top30 & hitter_idx)

    sr0, t0_ = metricas(np.zeros(M))
    sr390, t390 = metricas(kin[J390])
    print(f"\nreferencias: sin momento {sr0:+.3f} {t0_}/14 | solo z=390 {sr390:+.3f} {t390}/14")

    print(f"\n===== BARRIDO DE PLANOS: donde garantizar colimacion discrimina =====")
    print(f"  {'plano':<8} {'kin hit/todos':>14} {'separa':>7}  {'solo':>13}  {'combo c/z390':>13}")
    resultados = []
    for j in range(n_pl):
        kh, kt = kin[j][hit_mask].mean(), kin[j].mean()
        sr_solo, top_solo = metricas(kin[j])
        if j == J390:
            sr_c, top_c = sr_solo, top_solo
        else:
            sr_c, top_c = metricas(0.5 * (kin[j] + kin[J390]))
        resultados.append((ETIQ[j], kh, kt, kt - kh, sr_solo, top_solo, sr_c, top_c))
        marca = "  <- referencia" if j == J390 else (
            "  ** SUPERA a z390 solo" if top_c > t390 or (top_c == t390 and sr_c > sr390 + .005) else "")
        print(f"  {ETIQ[j]:<8} {kh:>6.3f}/{kt:<6.3f} {kt - kh:>+7.3f}  "
              f"{sr_solo:>+7.3f} {top_solo:>2}/14  {sr_c:>+7.3f} {top_c:>2}/14{marca}")

    print("\n  'separa' = kin(todos) - kin(hitters): cuanto mas positivo, mas")
    print("  distinguibles son los futuros hitters en ese plano. 'combo' =")
    print("  base + 20*promedio(kin(plano), kin(z390)).")

    mejores = sorted((r for r in resultados if r[0] != ETIQ[J390]),
                     key=lambda r: (-r[7], -r[6]))[:3]
    print("\n  mejores candidatos a segundo plano de KIN_PLANES:")
    for e, kh, kt, sep, srs, ts, src, tc in mejores:
        print(f"    {e}: combo {src:+.3f} {tc}/14 (separacion {sep:+.3f})")


if __name__ == "__main__":
    main()
