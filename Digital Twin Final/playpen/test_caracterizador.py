"""Tests del caracterizador unico (playpen: descartable).

1. Recuperacion de Twiss: haz sintetico 2D con matriz sigma conocida ->
   alfa/beta/emitancia deben recuperarse con error < 5%.
2. Residuo de transporte: si el estado final es un mapa LINEAL exacto del
   inicial, el residuo debe ser ~0; con un termino cuadratico agregado,
   debe crecer.
3. Los hits y distancias replican la regla de optimizer.count_hits.
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from caracterizador import caracterizar, hits_en_ventana, distancias_al_detector

rng = np.random.default_rng(7)
N = 5000

# ---------------------------------------------------------------- test 1
# haz gaussiano correlacionado con Twiss conocidos en x:
emit_obj = 2.0e-3        # mm*rad  (= 2.0 mm*mrad)
beta_obj = 8.0           # mm/rad
alfa_obj = -1.5          # divergiendo
gamma_obj = (1 + alfa_obj ** 2) / beta_obj
cov = emit_obj * np.array([[beta_obj, -alfa_obj], [-alfa_obj, gamma_obj]])
x, xp = rng.multivariate_normal([76.0, 0.0], cov, size=N).T

pos = np.column_stack([x, np.full(N, 76.0), np.full(N, 405.0)])
vz = np.full(N, 5.0e7)
vel = np.column_stack([xp * vz, np.zeros(N), vz])

f = caracterizar(pos, vel)
err_a = abs(f["twiss_alpha_x"] - alfa_obj) / abs(alfa_obj)
err_b = abs(f["twiss_beta_x"] - beta_obj) / beta_obj
err_e = abs(f["emittance_x"] - emit_obj * 1e3) / (emit_obj * 1e3)
print(f"T1 Twiss: alfa={f['twiss_alpha_x']:+.3f} (obj {alfa_obj:+.1f}, err {err_a*100:.1f}%)  "
      f"beta={f['twiss_beta_x']:.3f} (obj {beta_obj:.1f}, err {err_b*100:.1f}%)  "
      f"emit={f['emittance_x']:.3f} mm*mrad (obj {emit_obj*1e3:.1f}, err {err_e*100:.1f}%)")
assert err_a < 0.05 and err_b < 0.05 and err_e < 0.05, "Twiss no se recupera"

# ---------------------------------------------------------------- test 2
# transporte lineal exacto: x1 = 2*x0 + 30*x0' + 5, y1 = -y0 + 10*y0'
x0 = rng.normal(0.0, 1.0, N)
y0 = rng.normal(0.0, 1.0, N)
xp0 = rng.normal(0.0, 0.02, N)
yp0 = rng.normal(0.0, 0.02, N)
p0 = np.column_stack([x0 + 395.0, y0 + 75.0, np.full(N, 77.0)])
v0 = np.column_stack([xp0 * vz, yp0 * vz, vz])

x1 = 2 * (x0 + 395.0) + 30 * xp0 + 5 - 700.0    # centrado cerca de la ventana
y1 = -(y0 + 75.0) + 10 * yp0 + 151.0
p1 = np.column_stack([x1, y1, np.full(N, 405.0)])
v1 = vel.copy()

f_lin = caracterizar(p1, v1, p0, v0)
print(f"T2 residuo lineal: x={f_lin['resid_transporte_x_mm']:.2e} mm  "
      f"y={f_lin['resid_transporte_y_mm']:.2e} mm  (deben ser ~0)")
assert f_lin["resid_transporte_x_mm"] < 1e-8, "residuo no nulo en mapa lineal"

x1nl = x1 + 0.8 * (x0 ** 2)                      # aberracion cuadratica
f_nl = caracterizar(np.column_stack([x1nl, y1, np.full(N, 405.0)]), v1, p0, v0)
print(f"T2b con aberracion x^2: residuo x={f_nl['resid_transporte_x_mm']:.3f} mm (debe crecer)")
assert f_nl["resid_transporte_x_mm"] > 0.5, "el residuo no detecta la aberracion"

# ---------------------------------------------------------------- test 3
dentro = np.array([[76.0, 76.0, 405.0]])
fuera = np.array([[76.0, 76.0, 300.0]])
assert hits_en_ventana(dentro) == 1 and hits_en_ventana(fuera) == 0
assert distancias_al_detector(dentro)[0] == 0.0
assert abs(distancias_al_detector(fuera)[0] - 103.0) < 1e-9
print("T3 hits/distancias: OK")

print("\nTODOS LOS TESTS PASAN")
