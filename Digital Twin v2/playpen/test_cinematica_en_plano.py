"""
test_cinematica_en_plano.py
============================

Tests de caracterizador.cinematica_en_plano / puntaje_cinematico sobre
trayectorias sinteticas rectas (vuelo balistico: la interpolacion lineal
del cruce es EXACTA, asi que la recuperacion debe ser a precision de
maquina contra el calculo analitico x(z) = x0 + x'*(z - z0)).

Correr:  python playpen/test_cinematica_en_plano.py
"""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from caracterizador import cinematica_en_plano, puntaje_cinematico, _twiss_1d

Z0, Z_PLANO, VZ = 380.0, 390.0, 1.0e6   # mm, mm, mm/s
T, DT = 51, 4.0e-7                      # 51 pasos, cruza el plano ~a mitad


def trayectoria_recta(x0, y0, xp, yp, n_pasos=T, dt=DT):
    """(T,N,3) posiciones y velocidades de N particulas en vuelo recto."""
    n = len(x0)
    vel = np.stack([xp * VZ, yp * VZ, np.full(n, VZ)], axis=1)   # (N,3)
    pos0 = np.stack([x0, y0, np.full(n, Z0)], axis=1)
    t = np.arange(n_pasos)[:, None, None] * dt
    pos = pos0[None] + vel[None] * t
    velT = np.broadcast_to(vel[None], (n_pasos, n, 3)).copy()
    return pos, velT


def test_recuperacion_exacta():
    """div/alpha/emitancia en el plano == calculo analitico directo."""
    rng = np.random.default_rng(7)
    n = 80
    x0 = rng.normal(0.0, 2.0, n)
    y0 = rng.normal(0.0, 3.0, n)
    # haz convergente en x (xp anticorrelacionado con x), divergente en y
    xp = -0.004 * x0 + rng.normal(0, 0.002, n)
    yp = +0.006 * y0 + rng.normal(0, 0.003, n)

    pos, vel = trayectoria_recta(x0, y0, xp, yp)
    f = cinematica_en_plano(pos, vel, z_plano=Z_PLANO)

    # analitico: estado en el plano por propagacion balistica exacta
    x_pl = x0 + xp * (Z_PLANO - Z0)
    y_pl = y0 + yp * (Z_PLANO - Z0)
    a_x, _, e_x = _twiss_1d(x_pl, xp)
    a_y, _, e_y = _twiss_1d(y_pl, yp)

    assert f["n_cruzan"] == n, f["n_cruzan"]
    assert abs(f["div_x_mrad"] - np.std(xp) * 1e3) < 1e-6
    assert abs(f["div_y_mrad"] - np.std(yp) * 1e3) < 1e-6
    assert abs(f["twiss_alpha_x"] - a_x) < 1e-9
    assert abs(f["twiss_alpha_y"] - a_y) < 1e-9
    assert abs(f["emittance_x"] - e_x) < 1e-9
    assert abs(f["emittance_y"] - e_y) < 1e-9
    # convergente en x -> alpha > 0; divergente en y -> alpha < 0
    assert f["twiss_alpha_x"] > 0 and f["twiss_alpha_y"] < 0
    print(f"  recuperacion exacta OK (alpha_x={f['twiss_alpha_x']:+.2f}, "
          f"alpha_y={f['twiss_alpha_y']:+.2f}, emit_x={f['emittance_x']:.1f} mm*mrad)")


def test_pared_excluye():
    """Una particula que toca pared ANTES de cruzar no cuenta; una que
    toca DESPUES del cruce si cuenta."""
    rng = np.random.default_rng(3)
    n = 20
    pos, vel = trayectoria_recta(rng.normal(0, 2, n), rng.normal(0, 2, n),
                                 rng.normal(0, .003, n), rng.normal(0, .003, n))
    base = cinematica_en_plano(pos, vel, z_plano=Z_PLANO)["n_cruzan"]

    pared = np.full(n, -1)
    pared[:5] = 2          # tocan pared en el paso 2 (mucho antes del cruce ~25)
    pared[5:8] = T - 1     # tocan pared al final (despues del cruce)
    f = cinematica_en_plano(pos, vel, paso_pared=pared, z_plano=Z_PLANO)
    assert f["n_cruzan"] == base - 5, (f["n_cruzan"], base)
    print(f"  exclusion por pared OK ({base} -> {f['n_cruzan']} al matar 5 antes del cruce)")


def test_stop_idx_limita():
    """Particulas resueltas antes de llegar al plano no cuentan."""
    rng = np.random.default_rng(5)
    n = 12
    pos, vel = trayectoria_recta(rng.normal(0, 2, n), rng.normal(0, 2, n),
                                 rng.normal(0, .003, n), rng.normal(0, .003, n))
    stop = np.full(n, T - 1)
    stop[:6] = 3           # resueltas en el paso 3, antes de cruzar
    f = cinematica_en_plano(pos, vel, stop_idx=stop, z_plano=Z_PLANO)
    assert f["n_cruzan"] == n - 6
    print(f"  ventana stop_idx OK ({n} -> {f['n_cruzan']})")


def test_puntaje_ordena():
    """Haz bueno (colimado, cintura en el plano) << haz malo (divergente,
    emitancia grande); no medible = 1.0 exacto."""
    rng = np.random.default_rng(11)
    n = 100
    # bueno: 5 mrad de divergencia, sin correlacion (cintura en el plano)
    pos, vel = trayectoria_recta(rng.normal(0, 1, n), rng.normal(0, 1, n),
                                 rng.normal(0, .005, n), rng.normal(0, .005, n))
    bueno = puntaje_cinematico(cinematica_en_plano(pos, vel, z_plano=Z_PLANO))
    # malo: 150 mrad, fuertemente divergente
    x0, y0 = rng.normal(0, 4, n), rng.normal(0, 4, n)
    pos, vel = trayectoria_recta(x0, y0,
                                 .03 * x0 + rng.normal(0, .15, n),
                                 .03 * y0 + rng.normal(0, .15, n))
    malo = puntaje_cinematico(cinematica_en_plano(pos, vel, z_plano=Z_PLANO))
    # no medible: solo 2 particulas cruzan
    pos, vel = trayectoria_recta(np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2))
    nomed = puntaje_cinematico(cinematica_en_plano(pos, vel, z_plano=Z_PLANO))

    assert 0.0 <= bueno < 0.25, bueno
    assert malo > 2 * bueno, (bueno, malo)
    assert nomed == 1.0
    print(f"  puntaje ordena OK (bueno={bueno:.3f} < malo={malo:.3f}; no-medible=1.0)")


def test_cualquier_plano():
    """El mismo haz medido en dos planos: la divergencia (invariante en
    vuelo recto) se conserva; alpha cambia de signo al pasar la cintura."""
    rng = np.random.default_rng(13)
    n = 60
    x0 = rng.normal(0, 2, n)
    xp = -0.004 * x0 + rng.normal(0, 0.001, n)   # cintura en z ~ Z0+250mm... lejos
    y0, yp = rng.normal(0, 2, n), rng.normal(0, .002, n)
    pos, vel = trayectoria_recta(x0, y0, xp, yp, n_pasos=201)
    f1 = cinematica_en_plano(pos, vel, z_plano=385.0)
    f2 = cinematica_en_plano(pos, vel, z_plano=445.0)
    assert abs(f1["div_x_mrad"] - f2["div_x_mrad"]) < 1e-6   # invariante
    assert abs(f1["emittance_x"] - f2["emittance_x"]) < 1e-6  # invariante
    print(f"  medicion en cualquier plano OK (div y emitancia invariantes: "
          f"{f1['div_x_mrad']:.3f} mrad, {f1['emittance_x']:.2f} mm*mrad; "
          f"alpha 385mm={f1['twiss_alpha_x']:+.2f} vs 445mm={f2['twiss_alpha_x']:+.2f})")


def test_tramo_menos_x():
    """El tramo fuente->bender viaja en -x: eje=0, direccion=-1. Un haz
    identico volado en -x debe medir la MISMA divergencia que su gemelo
    volado en +z (las pendientes transversales son las mismas)."""
    rng = np.random.default_rng(17)
    n = 50
    y0, z0 = rng.normal(75, 2, n), rng.normal(77, 2, n)
    yp, zp = rng.normal(0, .004, n), rng.normal(0, .006, n)

    # vuelo en -x desde x=395: pos = (395 - vt, y0 + yp*..., z0 + zp*...)
    VX = 1.0e6
    vel = np.stack([np.full(n, -VX), yp * VX, zp * VX], axis=1)
    pos0 = np.stack([np.full(n, 395.0), y0, z0], axis=1)
    t = np.arange(T)[:, None, None] * DT
    pos = pos0[None] + vel[None] * t
    velT = np.broadcast_to(vel[None], (T, n, 3)).copy()

    f = cinematica_en_plano(pos, velT, z_plano=380.0, eje=0, direccion=-1)
    assert f["n_cruzan"] == n
    # claves x/y = transversales en orden (y, z); pendiente = v_t/v_x -> signo -
    assert abs(f["div_x_mrad"] - np.std(-yp) * 1e3) < 1e-6
    assert abs(f["div_y_mrad"] - np.std(-zp) * 1e3) < 1e-6
    print(f"  tramo -x OK (div transversales {f['div_x_mrad']:.2f} / "
          f"{f['div_y_mrad']:.2f} mrad recuperadas en plano x=380)")


if __name__ == "__main__":
    print("test_cinematica_en_plano:")
    test_recuperacion_exacta()
    test_pared_excluye()
    test_stop_idx_limita()
    test_puntaje_ordena()
    test_cualquier_plano()
    test_tramo_menos_x()
    print("TODOS LOS TESTS PASAN")
