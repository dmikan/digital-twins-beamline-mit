"""Tests del objetivo v2 normalizado (playpen: descartable).

1. Cotas: J en [0,1] (sin pared) para haces extremos.
2. Jerarquia: pasar de 0 a 100 hits debe mover J mas que arreglar
   CUALQUIER termino de forma completo (hits por encima de todo).
3. Regimenes: config que no llega -> J alto con gradiente solo por
   acercamiento; haz perfecto -> J ~ 0.
4. Escalera: el desglose respeta offset > halo > kurtosis > colim > twiss
   cuando todos estan en su peor valor.
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from caracterizador import objetivo_v2, PESOS_V2

BASE_PERFECTO = dict(n_considerados=500, hits=500, dist_punta_mm=0.0,
                     plane_fraction=1.0,
                     offset_x_mm=0.0, offset_y_mm=0.0, halo_fraction=0.0,
                     kurtosis_x=0.0, kurtosis_y=0.0, div_x_mrad=0.0,
                     div_y_mrad=0.0, twiss_alpha_x=0.0, twiss_alpha_y=0.0)

# ------------------------------------------------------------------ T1/T3
J_perf, d_perf = objetivo_v2(BASE_PERFECTO)
# El termino de transmision SATURANTE nunca llega a 0 exacto: con 500
# hits vale 0.5/(1+500/15) ~= 0.0146 (piso desde el fix v2.2/v2.3; este
# assert estaba desactualizado desde entonces -- el termino lineal v2.4
# si es 0 exacto con hits=n_total).
print(f"T1 haz perfecto: J={J_perf:.4f} (debe ser ~0.005, piso saturante/3)")
assert J_perf < 0.02
assert d_perf.get("transmision_lineal", 0.0) < 1e-9

f_nada = dict(n_considerados=500, hits=0, dist_punta_mm=330.0,
              plane_fraction=0.0)  # no llega nada
J_nada, d_nada = objetivo_v2(f_nada)
# v2.4 re-normaliza por (1+lambda): la cota [0,1] sin pared se conserva
print(f"T1 config perdido: J={J_nada:.4f} (alto, <=1)")
assert 0.9 < J_nada <= 1.0

f_cerca = dict(f_nada, dist_punta_mm=20.0)
J_cerca, _ = objetivo_v2(f_cerca)
print(f"T3 gradiente lejos: d=330 -> J={J_nada:.4f}; d=20 -> J={J_cerca:.4f} (debe bajar)")
assert J_cerca < J_nada - 0.016, "el acercamiento no da gradiente"  # 0.05/3 (escala v2.4)

# ------------------------------------------------------------------ T2
# jerarquia: 100 hits (de 500) vs arreglar por completo el mejor termino
# de forma partiendo del mismo estado
f_llega_mal = dict(n_considerados=500, hits=0, dist_punta_mm=0.0,
                   plane_fraction=0.3,
                   offset_x_mm=6.0, offset_y_mm=6.5, halo_fraction=0.3,
                   kurtosis_x=6.0, kurtosis_y=6.0, div_x_mrad=125.0,
                   div_y_mrad=125.0, twiss_alpha_x=1.4, twiss_alpha_y=0.9)
J0, _ = objetivo_v2(f_llega_mal)
J_hits, _ = objetivo_v2(dict(f_llega_mal, hits=100))
J_offset, _ = objetivo_v2(dict(f_llega_mal, offset_x_mm=0.0, offset_y_mm=0.0))
mejora_hits = J0 - J_hits
mejora_offset = J0 - J_offset
print(f"T2 mejora por +100 hits: {mejora_hits:.4f}  vs arreglar offset completo: {mejora_offset:.4f}")
assert mejora_hits > mejora_offset, "hits no domina sobre el offset"

# T2c: el termino de CUERPO da gradiente entre configs con 0 hits --
# el regimen medido el 2026-07-05 (375/500 al plano con 0 hits vs nada)
J_cuerpo_alto, _ = objetivo_v2(dict(f_llega_mal, plane_fraction=0.75))
print(f"T2c cuerpo: plane 0.30 -> J={J0:.4f}; plane 0.75 -> J={J_cuerpo_alto:.4f} (debe bajar)")
assert J_cuerpo_alto < J0 - 0.01, "el termino de cuerpo no da gradiente"  # 0.03/3 (escala v2.4)

# ------------------------------------------------------------------ T4
_, d_peor = objetivo_v2(dict(n_considerados=500, hits=0, dist_punta_mm=1e6))
escalera = ["offset", "halo", "kurtosis", "colimacion", "twiss"]
vals = [d_peor[k] for k in escalera]
print("T4 escalera en peor caso:", {k: round(v, 3) for k, v in zip(escalera, vals)})
assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)), "escalera desordenada"
assert PESOS_V2["transmision"] > sum(PESOS_V2[k] for k in escalera), \
    "hits no esta por encima de la suma de la forma"

# ------------------------------------------------------------------ T5
# Regimen ALTO de hits (v2.4, regresion del bug entre-cuencas 2026-07-10):
# con la transmision saturante sola (lambda=0), pasar de 84 a 105 hits
# valia ~0.013 de J -- menos que la brecha cosmetica real medida entre
# la cuenca Search v2.2 (84 hits, J=0.170) y la TES-GP-GP6D (105 hits,
# J=0.227), ~0.066 en offset+twiss+colim+halo. En la escala v2.4
# (re-normalizada por 1+lambda=3) esa brecha vale ~0.022; los 21 hits
# deben valer mas que eso (~0.032 con lambda=2).
f_alto = dict(n_total=500, n_considerados=500, hits=84, dist_punta_mm=0.0,
              plane_fraction=1.0)
J_84, _ = objetivo_v2(f_alto)
J_105, _ = objetivo_v2(dict(f_alto, hits=105))
mejora_alto = J_84 - J_105
print(f"T5 regimen alto: 84->105 hits mejora J en {mejora_alto:.4f} (debe ser >=0.027)")
assert mejora_alto >= 0.027, "v2.4: +21 hits no domina la brecha cosmetica entre cuencas"

J_84_v23, _ = objetivo_v2(f_alto, lambda_hits=0)
J_105_v23, _ = objetivo_v2(dict(f_alto, hits=105), lambda_hits=0)
print(f"T5b sanity: con lambda=0 la mejora era solo {J_84_v23 - J_105_v23:.4f} (el bug)")
assert (J_84_v23 - J_105_v23) < 0.022, "si esto falla, el termino saturante cambio"

print("\nTODOS LOS TESTS PASAN")
