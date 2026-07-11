"""
medir_desalineacion_stl.py (2026-07-06)
=======================================

Cuantifica el desalineamiento STL-vs-realidad usando los mapas de metal
del PA (basis_quad/mask_*.csv) como VERDAD -- estan en coordenadas
workbench, la MISMA fuente que el campo. Antes esto era una estimacion a
mano ("~11-15mm, 6 landmarks del cursor"); ahora hay ~150k puntos de
metal reales para medirlo.

Mide:
  (a) distancia bidireccional STL<->PA metal (Chamfer): cuanto se
      aparta la superficie STL del metal verdadero.
  (b) offset del centroide por electrodo del cuadrupolo (9-12): revela
      el corrimiento rigido aunque el metal interior lo oculte en (a).

Correr:  python playpen/medir_desalineacion_stl.py
"""

import pathlib
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import physics as phys


def main():
    # verdad: metal del PA (workbench)
    pa = np.loadtxt(ROOT / "basis_quad" / "mask_global_2mm.csv", delimiter=",", skiprows=1)
    pa_tree = cKDTree(pa)
    print(f"PA metal (verdad): {len(pa)} puntos @2mm")

    # STL transformados con la matriz actual
    meshes = phys.load_electrode_meshes(ROOT)
    excl = (1, 2, 19)
    surf = []
    for e, m in meshes.items():
        if e in excl:
            continue
        s, _ = m.sample(3000, return_index=True)
        surf.append(s)
    surf = np.vstack(surf)
    surf_tree = cKDTree(surf)

    # (a) Chamfer bidireccional
    d_stl_to_pa, _ = pa_tree.query(surf, workers=-1)     # superficie STL -> metal real
    d_pa_to_stl, _ = surf_tree.query(pa, workers=-1)      # metal real -> superficie STL
    print("\n(a) DISTANCIA STL <-> METAL PA (mm):")
    print(f"    STL->PA : mediana {np.median(d_stl_to_pa):5.1f}  p90 {np.percentile(d_stl_to_pa,90):5.1f}  max {d_stl_to_pa.max():5.1f}")
    print(f"    PA->STL : mediana {np.median(d_pa_to_stl):5.1f}  p90 {np.percentile(d_pa_to_stl,90):5.1f}  max {d_pa_to_stl.max():5.1f}")
    print(f"    (alineacion perfecta daria ~2mm = espaciado de grilla)")

    # (b) offset de centroides del cuadrupolo (corrimiento rigido)
    quad_pa = np.loadtxt(ROOT / "basis_quad" / "mask_quad_1mm.csv", delimiter=",", skiprows=1)
    print("\n(b) OFFSET DE CENTROIDE POR ELECTRODO DEL QUAD (9-12):")
    print(f"    {'elec':>4} {'centroide STL':>22} {'centroide PA':>22} {'offset mm':>9}")
    # clasificar cada punto PA del quad por el electrodo STL mas cercano
    centros_stl = {e: meshes[e].vertices.mean(axis=0) for e in (9, 10, 11, 12)}
    arboles = {e: cKDTree(meshes[e].sample(2000)) for e in (9, 10, 11, 12)}
    dist_e = np.stack([arboles[e].query(quad_pa, workers=-1)[0] for e in (9, 10, 11, 12)])
    duenio = np.array([9, 10, 11, 12])[np.argmin(dist_e, axis=0)]
    offsets = []
    for e in (9, 10, 11, 12):
        pa_e = quad_pa[duenio == e]
        c_pa = pa_e.mean(axis=0)
        c_stl = centros_stl[e]
        off = np.linalg.norm(c_stl - c_pa)
        offsets.append(off)
        print(f"    {e:>4} ({c_stl[0]:5.0f},{c_stl[1]:5.0f},{c_stl[2]:5.0f})   "
              f"({c_pa[0]:5.0f},{c_pa[1]:5.0f},{c_pa[2]:5.0f})   {off:>9.1f}")
    print(f"\n    offset medio del quad: {np.mean(offsets):.1f} mm")
    print(f"    => si es >~3mm, la STL esta desalineada; el metal del PA")
    print(f"       (ParedesPA) evita el problema por completo.")


if __name__ == "__main__":
    main()
