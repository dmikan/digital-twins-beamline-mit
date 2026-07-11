"""
barrido_angulo.py -- techo fisico de transmision (2026-07-07)
=============================================================

Metodo 1: barre el SEMIANGULO del cono de la fuente sobre el mejor config
y mide la transmision. Como la transmision satura al achicar el angulo,
la curva revela el semiangulo de ACEPTANCIA (donde las aperturas empiezan
a recortar) y si el config actual esta cerca del techo fisico.

Se hace en el twin RK4 porque el haz de SIMION esta fijo en el .iob (no
se controla el cono desde Python), pero el twin usa los campos y paredes
REALES del PA -> su geometria de aperturas es fiel, que es lo que importa
para el techo. Se ancla al punto SIMION medido a 15 grados (~23%).

Config base = maxima transmision del registro (estable).
Salida: outputs/report_figures/fig_barrido_angulo.png
Correr:  python playpen/barrido_angulo.py
"""

import json
import pathlib
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import optimizer as op
import physics as phys
from caracterizador import make_beam, BeamProgressScorer

SPECIES = phys.IonSpecies(mass=28 * 1.66053906660e-27, charge=1.602176634e-19)
ANGULOS = [15.0, 12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.1]
N_PART = 500
PASOS = 3000
DT = 5e-9
SIMION_15 = 0.23   # transmision SIMION medida a 15 grados (ancla)


def config_record():
    reg = ROOT / "studies" / "registro_corridas.jsonl"
    filas = [json.loads(l) for l in reg.read_text().splitlines() if l.strip()]
    sim = [f for f in filas if f.get("fuente") == "simion" and f.get("hits") is not None and f.get("voltajes")]
    mejor = max(sim, key=lambda f: f["hits"])
    v = np.zeros(19)
    for e, val in op.FIXED.items():
        v[e - 1] = val
    for k, val in mejor["voltajes"].items():
        e = int(str(k).lstrip("V"))
        if e not in op.FIXED:
            v[e - 1] = float(val)
    return v, mejor["hits"]


def main():
    base, hits_ref = config_record()
    print(f"Config base (registro): {hits_ref} hits @ 15 grados en SIMION\n")
    fis = phys.cargar_fisica(ROOT)
    fis.bfm.set_voltages_batch(base[None, :])

    T = []
    for th in ANGULOS:
        sp, sv = make_beam(N=N_PART, species=SPECIES, start_point=[395, 75, 77],
                           mean_energy_eV=15, std_energy_eV=0.42466, half_angle_deg=th, seed=42)
        beam, ci = phys.make_batch_beam(SPECIES, sp, sv, 1)
        traj = phys.BatchTrajectory(beam, ci)
        phys.BatchRK4Integrator(fis.bfm, ci).integrate(traj, dt=DT, num_steps=PASOS)
        sc = BeamProgressScorer(bfm=fis.bfm, Trajectory=traj, dt=DT, num_steps=PASOS,
            detector_bbox=op.DETECTOR_BBOX, wall_index=fis.wall, wall_hit_margin=fis.margen,
            wall_check_midpoints=False, wall_check_stride=3)
        r = sc.score(base[None, :])
        t = float(r["reached_target"].mean())
        T.append(t)
        print(f"  half-angle={th:>5.1f} deg  ->  transmision twin = {t*100:.1f}%")
    T = np.array(T)

    # semiangulo de aceptancia: donde T cae a la mitad de su maximo (a angulo chico)
    Tmax = T.max()
    idx_acc = None
    for i in range(len(ANGULOS) - 1, -1, -1):
        if T[i] >= 0.5 * Tmax:
            idx_acc = i
    print(f"\n  transmision twin maxima (angulo->0): {Tmax*100:.0f}%")
    print(f"  a 15 grados: twin {T[0]*100:.0f}%  vs  SIMION {SIMION_15*100:.0f}% (ancla)")
    if idx_acc is not None:
        print(f"  semiangulo de aceptancia (~mitad del maximo): ~{ANGULOS[idx_acc]:.0f} grados")
    frac_techo = T[0] / Tmax if Tmax > 0 else 0
    print(f"  el config a 15 grados capta {frac_techo*100:.0f}% de su propio techo (angulo->0)")

    # figura
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot(ANGULOS, T * 100, "o-", color="tab:blue", label="transmision twin (RK4)")
    ax.axhline(SIMION_15 * 100, color="tab:red", ls="--", lw=1,
               label=f"SIMION @15 grados = {SIMION_15*100:.0f}% (ancla)")
    ax.axvline(15, color="0.6", ls=":", lw=1, label="operacion real (15 grados)")
    ax.set_xlabel("semiangulo del cono de la fuente (grados)")
    ax.set_ylabel("transmision al detector (%)")
    ax.set_title("Techo fisico de transmision: barrido de semiangulo\n"
                 f"(mejor config del registro, {hits_ref} hits @15 grados)")
    ax.legend(); ax.grid(alpha=0.3); ax.invert_xaxis()
    fig.tight_layout()
    out = ROOT / "outputs" / "report_figures" / "fig_barrido_angulo.png"
    fig.savefig(out, dpi=140)
    print(f"\nFigura: {out}")


if __name__ == "__main__":
    main()
