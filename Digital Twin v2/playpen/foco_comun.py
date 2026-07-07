"""
foco_comun.py -- maquinaria compartida de los barridos de enfoque
(scan_placas.py, scan_forma_quad.py). Fisica promovida (dual 1mm),
config base = mejor del estudio gemelo_v2 (familia record), metricas de
enfoque en el plano pre-detector + proxy de hits (entrada a la caja).
"""

import pathlib
import sys
import time

import numpy as np
import optuna

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

optuna.logging.set_verbosity(optuna.logging.WARNING)

from physics import IonSpecies
from physics import make_batch_beam, BatchTrajectory, BatchRK4Integrator
from caracterizador import make_beam, BeamProgressScorer
from caracterizador import cinematica_en_plano, DET_CENTER
import optimizer as orch
orch.N_ELECTRODES = 19
from physics import cargar_fisica

N_PART, PASOS, DT, CHUNK = 200, 3000, 1.0e-8, 8
QUAD = (9, 10, 11, 12)


def base_del_estudio():
    """Config base = mejor por HITS del estudio gemelo_v2 (el record de
    transmision), NO por J_v2 -- que rankea configs de 0 hits arriba del
    record (medido 2026-07-06). Para enfoque hay que partir del que pega."""
    from gemelo import GemeloDigital
    tw = GemeloDigital()
    m = tw.mejor(por="hits")
    if m is None:
        raise SystemExit("Estudio sin trials con hits -- corre playpen/sembrar_record.py.")
    print(f"Base: mejor POR HITS (trial {m['trial']}, hits={m['hits']}, "
          f"J={m['objetivo']:.3f}): {m['voltajes']}")
    return tw.voltajes_completos(m["voltajes"])


def volar_configs(volts, fis=None, species=None):
    """Vuela (M,19) configs con la fisica promovida y devuelve, por
    config: reach (proxy de hits: fraccion que entra a la caja del
    detector), n limpias / centroide / sigma en z=390."""
    fis = fis or cargar_fisica(ROOT)
    species = species or IonSpecies(mass=28 * 1.66053906660e-27, charge=1.602176634e-19)
    sp, sv = make_beam(N=N_PART, species=species, start_point=[395.0, 75.0, 77.0],
                       mean_energy_eV=15.0, std_energy_eV=0.42466, half_angle_deg=15.0, seed=42)
    out = []
    t0 = time.time()
    for lo in range(0, len(volts), CHUNK):
        chunk = volts[lo:lo + CHUNK]
        n_cfg = chunk.shape[0]
        fis.bfm.set_voltages_batch(chunk)
        beam, ci = make_batch_beam(species, sp, sv, n_cfg)
        traj = BatchTrajectory(beam, ci)
        BatchRK4Integrator(fis.bfm, ci).integrate(traj, dt=DT, num_steps=PASOS)
        scorer = BeamProgressScorer(
            bfm=fis.bfm, Trajectory=traj, dt=DT, num_steps=PASOS,
            detector_bbox=orch.DETECTOR_BBOX, wall_index=fis.wall,
            wall_hit_margin=fis.margen, wall_check_midpoints=False, wall_check_stride=3)
        r = scorer.score(chunk)
        pos = r["positions"].reshape(-1, n_cfg, N_PART, 3)
        stop = r["stop_idx"].reshape(n_cfg, N_PART)
        pared = r["hit_wall_step"].reshape(n_cfg, N_PART)
        reach = r["reached_target"].reshape(n_cfg, N_PART).mean(axis=1)
        for c in range(n_cfg):
            vel_c = np.asarray([s.velocity[c * N_PART:(c + 1) * N_PART] for s in traj.states],
                               dtype=np.float32)
            f = cinematica_en_plano(pos[:, c], vel_c, stop_idx=stop[c],
                                    paso_pared=pared[c], z_plano=390.0)
            off_x = f["centro_x_mm"] - DET_CENTER[0]
            off_y = f["centro_y_mm"] - DET_CENTER[1]
            out.append(dict(reach=float(reach[c]), n390=f["n_cruzan"],
                            off_x=float(off_x), off_y=float(off_y),
                            sigma_x=f["sigma_x_mm"], sigma_y=f["sigma_y_mm"]))
        print(f"  configs {lo}-{lo + n_cfg - 1}  ({time.time() - t0:.0f}s)")
    return out


def tabla(nombres, metricas, top=12):
    """Imprime el resumen ordenado por reach (proxy de hits) descendente."""
    orden = np.argsort([-m["reach"] for m in metricas])
    print(f"\n  {'config':<28} {'reach':>6} {'n390':>5} {'off_x':>6} {'off_y':>6} "
          f"{'sig_x':>6} {'sig_y':>6}")
    for i in orden[:top]:
        m = metricas[i]
        print(f"  {nombres[i]:<28} {m['reach']:>6.3f} {m['n390']:>5} "
              f"{m['off_x']:>+6.1f} {m['off_y']:>+6.1f} "
              f"{m['sigma_x']:>6.1f} {m['sigma_y']:>6.1f}")
    return orden
