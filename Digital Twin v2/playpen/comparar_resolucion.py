"""
comparar_resolucion.py -- 2.5mm vs 1mm, controlado (2026-07-06)
===============================================================

UNA sola comparacion apples-to-apples para decidir si revertir a 2.5mm:
  - mismas cajas anchas (solo cambia la resolucion del campo)
  - mismo muro (ParedesPA, cargado una vez)
  - mismo dataset ESTABLE: las 69 corridas SIMION del registro (voltajes
    + hits conocidos), inmune a la fluctuacion de las DBs archivadas
  - mismo margen (0.5), misma fidelidad

Reporta las DOS cosas que importan:
  (a) ranking: Spearman(score RK4, hits) + hitters en top-15
  (b) live/dead: n limpias en z390 de los 5 casos canonicos vs SIMION

Correr:  python playpen/comparar_resolucion.py
"""

import json
import pathlib
import sys
import time

import numpy as np
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import physics as phys
import optimizer as op
import caracterizador as carac

QUAD = (9, 10, 11, 12)
SPECIES = phys.IonSpecies(mass=28 * 1.66053906660e-27, charge=1.602176634e-19)


def cargar_dataset():
    """69 configs del registro: voltajes(19), hits."""
    reg = ROOT / "studies" / "registro_corridas.jsonl"
    filas = [json.loads(l) for l in reg.read_text().splitlines() if l.strip()]
    sim = [f for f in filas if f.get("fuente") == "simion"
           and f.get("hits") is not None and f.get("voltajes")]
    volts, hits = [], []
    for f in sim:
        v = np.zeros(19)
        for e, val in op.FIXED.items():
            v[e - 1] = val
        for k, val in f["voltajes"].items():
            e = int(str(k).lstrip("V"))
            if e not in op.FIXED:
                v[e - 1] = float(val)
        volts.append(v)
        hits.append(f["hits"])
    return np.array(volts), np.array(hits)


def canonicos(base):
    def con(v3=None, esc=None, v6=None):
        v = base.copy()
        if v3 is not None: v[2] = v3
        if esc is not None:
            for e in QUAD: v[e - 1] = base[e - 1] * esc
        if v6 is not None: v[5] = v6
        return v
    return [("BASE", con(), 51), ("x1.10", con(esc=1.10), 0),
            ("250/x1.25", con(250.0, 1.25), 0),
            ("250/x0.70", con(250.0, 0.70), 368),
            ("RECORD", con(250.0, 0.70, -750.0), 384)]


def n_limpias_z390(bfm, wall, volts, particulas=200, pasos=3000):
    sp, sv = carac.make_beam(N=particulas, species=SPECIES, start_point=[395, 75, 77],
                             mean_energy_eV=15, std_energy_eV=0.42466, half_angle_deg=15, seed=42)
    out = []
    for lo in range(0, len(volts), 8):
        chunk = volts[lo:lo + 8]; n = chunk.shape[0]
        bfm.set_voltages_batch(chunk)
        beam, ci = phys.make_batch_beam(SPECIES, sp, sv, n)
        traj = phys.BatchTrajectory(beam, ci)
        phys.BatchRK4Integrator(bfm, ci).integrate(traj, dt=1e-8, num_steps=pasos)
        sc = carac.BeamProgressScorer(bfm=bfm, Trajectory=traj, dt=1e-8, num_steps=pasos,
            detector_bbox=op.DETECTOR_BBOX, wall_index=wall, wall_hit_margin=0.5,
            wall_check_midpoints=False, wall_check_stride=3)
        r = sc.score(chunk)
        pos = r["positions"].reshape(-1, n, particulas, 3)
        stop = r["stop_idx"].reshape(n, particulas)
        pared = r["hit_wall_step"].reshape(n, particulas)
        for c in range(n):
            vel_c = np.asarray([s.velocity[c*particulas:(c+1)*particulas] for s in traj.states], dtype=np.float32)
            f = carac.cinematica_en_plano(pos[:, c], vel_c, stop_idx=stop[c], paso_pared=pared[c], z_plano=390.0)
            out.append(f["n_cruzan"] * (500 / particulas))
    return out


def ranking(bfm, wall, volts, pasos=1500):
    sp, sv = carac.make_beam(N=50, species=SPECIES, start_point=[395, 75, 77],
                             mean_energy_eV=15, std_energy_eV=0.42466, half_angle_deg=15, seed=42)
    scores = op.rk4_score_all(bfm, wall, volts, sp, sv, SPECIES, 1e-8, pasos, 8)
    return scores


def evaluar_resolucion(nombre_res, cajas, volts, hits, casos):
    print(f"\n----- {nombre_res} -----")
    gruesos = [ROOT / f"basis_electrode_{i}.csv" for i in range(1, 20)]
    t0 = time.time()
    bfm = phys.Campo1mm(gruesos, cajas, root=ROOT)
    wall = phys.ParedesPA.desde_proyecto(ROOT, verbose=False)
    print(f"  (carga {time.time()-t0:.0f}s)")

    # (b) live/dead
    volts_c = np.array([v for _, v, _ in casos])
    nl = n_limpias_z390(bfm, wall, volts_c)
    # (a) ranking
    t0 = time.time()
    sc = ranking(bfm, wall, volts)
    sr, _ = spearmanr(-sc, hits)
    top15 = set(int(i) for i in np.argsort(sc)[:15])
    hitters = set(int(i) for i in np.where(hits > 0)[0])
    print(f"  ranking {len(volts)} configs en {time.time()-t0:.0f}s")
    del bfm, wall
    return dict(spearman=sr, top15=len(top15 & hitters), n_hitters=len(hitters), nl=nl)


def main():
    volts, hits = cargar_dataset()
    base = volts[int(np.argmax(hits))]
    casos = canonicos(base)
    print(f"dataset: {len(volts)} configs del registro ({int((hits>0).sum())} con hits, "
          f"max {hits.max()})")

    cajas_1mm = [(n, [ROOT/"basis_quad"/f"basis_{n}_electrode_{i}.csv" for i in range(1,20)])
                 for n in ("quad", "c1", "c2")]
    cajas_25 = [(n, [ROOT/"basis_quad"/f"basis_{n}_electrode_{i}.csv" for i in range(1,20)])
                for n in ("quad25", "c125", "c225")]

    res1 = evaluar_resolucion("1mm", cajas_1mm, volts, hits, casos)
    res25 = evaluar_resolucion("2.5mm", cajas_25, volts, hits, casos)

    print("\n================ VEREDICTO ================")
    print(f"(a) RANKING (Spearman / hitters top-15 de {res1['n_hitters']}):")
    print(f"    1mm  : {res1['spearman']:+.3f}   {res1['top15']}/{res1['n_hitters']}")
    print(f"    2.5mm: {res25['spearman']:+.3f}   {res25['top15']}/{res25['n_hitters']}")
    print(f"\n(b) LIVE/DEAD (n limpias z390, escala 500):")
    print(f"    {'caso':<11} {'1mm':>7} {'2.5mm':>7} {'SIMION':>7}")
    for (nombre, _, verdad), a, b in zip(casos, res1["nl"], res25["nl"]):
        print(f"    {nombre:<11} {a:>7.0f} {b:>7.0f} {verdad:>7}")


if __name__ == "__main__":
    main()
