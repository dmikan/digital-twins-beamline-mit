"""
evaluacion_tareas.py -- Direccion + Consigna inversa (2026-07-07)
=================================================================

Dos evaluaciones del gemelo pedidas por la consigna del hackathon:

TAREA A -- DIRECCION (maximizar transmision = hits en el detector):
  el gemelo (optimizacion twin-guiada) propone los voltajes de maxima
  transmision; se predice con el twin, se corre 1 vez en SIMION y se
  reporta la fraccion transmitida real.

TAREA B -- CONSIGNA INVERSA (llevar el CENTROIDE del haz a un objetivo):
  se usa el twin (RK4, barato) para medir la respuesta local de los
  deflectores V15<->offset_x y V18<->offset_y alrededor del mejor config,
  se invierte para un offset objetivo, se corre 1 vez en SIMION y se mide
  |real - objetivo|.

Correr:  python playpen/evaluacion_tareas.py
"""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import optimizer as op
import physics as phys
from gemelo import GemeloDigital
from caracterizador import make_beam, BeamProgressScorer, cinematica_en_plano, DET_CENTER

SPECIES = phys.IonSpecies(mass=28 * 1.66053906660e-27, charge=1.602176634e-19)
OBJETIVO_OFFSET = (0.0, 0.0)   # consigna: centrar el haz en la ventana
N_VUELOS_A = 3                 # promedio para dar transmision +- ruido (Tarea A)


def mejor_global():
    """Mejor config por hits entre los dos estudios del gemelo."""
    best = None
    for est, db in (("gemelo_v2", "studies/gemelo_v2.db"),
                    ("gemelo_db_v2", "studies/gemelo_db_v2.db")):
        tw = GemeloDigital(estudio=est, db=ROOT / db)
        m = tw.mejor(por="hits")
        if m and (best is None or (m["hits"] or 0) > (best[1]["hits"] or 0)):
            best = (tw, m)
    return best


def twin_offsets(fis, volts, particulas=200, pasos=3000, dt=5e-9):
    """Offset del centroide en z=390 predicho por el twin, por config."""
    volts = np.atleast_2d(volts)
    n = len(volts)
    sp, sv = make_beam(N=particulas, species=SPECIES, start_point=[395, 75, 77],
                       mean_energy_eV=15, std_energy_eV=0.42466, half_angle_deg=15, seed=1234)
    fis.bfm.set_voltages_batch(volts)
    beam, ci = phys.make_batch_beam(SPECIES, sp, sv, n)
    traj = phys.BatchTrajectory(beam, ci)
    phys.BatchRK4Integrator(fis.bfm, ci).integrate(traj, dt=dt, num_steps=pasos)
    sc = BeamProgressScorer(bfm=fis.bfm, Trajectory=traj, dt=dt, num_steps=pasos,
        detector_bbox=op.DETECTOR_BBOX, wall_index=fis.wall, wall_hit_margin=fis.margen,
        wall_check_midpoints=False, wall_check_stride=3)
    r = sc.score(volts)
    pos = r["positions"].reshape(-1, n, particulas, 3)
    stop = r["stop_idx"].reshape(n, particulas)
    pared = r["hit_wall_step"].reshape(n, particulas)
    out = []
    for c in range(n):
        vel_c = np.asarray([s.velocity[c * particulas:(c + 1) * particulas] for s in traj.states],
                           dtype=np.float32)
        f = cinematica_en_plano(pos[:, c], vel_c, stop_idx=stop[c], paso_pared=pared[c], z_plano=390.0)
        # (offset_x, offset_y, n_particulas_al_plano): n habilita rechazar
        # pasos que matan la transmision (centroide sin sentido)
        out.append((f["centro_x_mm"] - DET_CENTER[0], f["centro_y_mm"] - DET_CENTER[1],
                    int(f["n_cruzan"] or 0)))
    return out


def inverse_twin(fis, base, target, iters=6, delta=60.0, tol=0.15, n_min=25):
    """
    Consigna inversa iterada SOLO con el twin (barato): Gauss-Newton
    amortiguado con LINE SEARCH. En cada paso mide el Jacobiano 2x2
    (dO/dV por diferencias finitas) y prueba pasos de tamano decreciente,
    aceptando el PRIMERO que (a) reduce el residuo Y (b) mantiene >=n_min
    particulas al plano (si no, el centroide no tiene sentido). Si ningun
    paso mejora, para -- ese es el offset ALCANZABLE (el centroide no es
    libremente controlable: mover los deflectores tambien afecta la
    transmision). Converge donde el twin predice; el residuo que mida
    SIMION despues es la brecha twin<->SIMION pura.
    """
    target = np.array(target, dtype=float)
    v15, v18 = float(base[14]), float(base[17])

    def cfg(a, b):
        c = base.copy(); c[14], c[17] = a, b
        return c

    O0 = np.array(twin_offsets(fis, cfg(v15, v18)[None])[0][:2])
    for it in range(iters):
        r = float(np.hypot(*(target - O0)))
        print(f"    iter {it}: twin offset=({O0[0]:+.2f},{O0[1]:+.2f}) "
              f"|resid|={r:.2f}  V15={v15:.0f} V18={v18:.0f}")
        if r < tol:
            break
        # Jacobiano (2 vuelos batch)
        jac = twin_offsets(fis, np.array([cfg(v15 + delta, v18), cfg(v15, v18 + delta)]))
        Ox, Oy = np.array(jac[0][:2]), np.array(jac[1][:2])
        J = np.column_stack([(Ox - O0) / delta, (Oy - O0) / delta])
        dV = np.linalg.lstsq(J, target - O0, rcond=None)[0]
        # line search: probar alphas decrecientes en un solo batch
        alphas = [1.0, 0.5, 0.25, 0.1, 0.05]
        cands = [cfg(float(np.clip(v15 + a * dV[0], -1000, 1000)),
                     float(np.clip(v18 + a * dV[1], -1000, 1000))) for a in alphas]
        res = twin_offsets(fis, np.array(cands))
        mejorado = False
        for a, cand, (ox, oy, n) in zip(alphas, cands, res):
            if n >= n_min and np.hypot(target[0] - ox, target[1] - oy) < r:
                v15, v18, O0 = float(cand[14]), float(cand[17]), np.array([ox, oy])
                mejorado = True
                break
        if not mejorado:
            print("    (ningun paso mejora sin matar la transmision -> optimo alcanzable)")
            break
    return v15, v18, O0


def main():
    tw, m = mejor_global()
    fis = phys.cargar_fisica(ROOT)
    tw._bfm, tw._wall, tw._margen = fis.bfm, fis.wall, fis.margen  # compartir fisica
    base = tw.voltajes_completos(m["voltajes"])

    # ================= TAREA A: DIRECCION (max transmision) =================
    print("\n" + "=" * 62)
    print("TAREA A -- DIRECCION: maximizar transmision (hits)")
    print("=" * 62)
    print(f"Voltajes del gemelo: {m['voltajes']}")
    pred = tw.predecir(m["voltajes"])
    print(f"  Prediccion del gemelo (RK4): reach={pred['reach_fraction']:.2f}, "
          f"J_v2={pred['objetivo']:.3f}")
    hits = [tw.evaluar(m["voltajes"])["hits"] for _ in range(N_VUELOS_A)]
    hits = np.array(hits, dtype=float)
    print(f"  SIMION real ({N_VUELOS_A} corridas): hits = {[int(h) for h in hits]}")
    print(f"  transmision = {hits.mean():.0f}+-{hits.std():.0f} / 500 = "
          f"{hits.mean()/500*100:.1f}% +- {hits.std()/500*100:.1f}%")
    print(f"  fraccion del optimo conocido: {hits.mean():.0f}/{m['hits']} = "
          f"{hits.mean()/max(1,m['hits'])*100:.0f}% (optimo previo registrado: {m['hits']})")

    # ================= TAREA B: CONSIGNA INVERSA (posicion) =================
    print("\n" + "=" * 62)
    print(f"TAREA B -- CONSIGNA INVERSA: centroide objetivo = {OBJETIVO_OFFSET} mm")
    print("=" * 62)
    v15_star, v18_star, O_twin = inverse_twin(fis, base, OBJETIVO_OFFSET)
    print(f"  Twin convergio: predice ({O_twin[0]:+.2f}, {O_twin[1]:+.2f}) mm "
          f"con V15={v15_star:.0f}, V18={v18_star:.0f}")

    cfg = base.copy(); cfg[14] = v15_star; cfg[17] = v18_star
    realB = tw.evaluar(cfg)
    fB = realB.get("features") or {}
    ax, ay = fB.get("offset_x_mm", np.nan), fB.get("offset_y_mm", np.nan)
    err = np.hypot(ax - OBJETIVO_OFFSET[0], ay - OBJETIVO_OFFSET[1])
    print(f"  SIMION real (1 corrida): offset=({ax:+.1f}, {ay:+.1f}) mm, hits={realB['hits']}")
    print(f"  |real - objetivo| = {err:.2f} mm  (= brecha twin<->SIMION, "
          f"el twin ya predecia ~el objetivo)")


if __name__ == "__main__":
    main()
