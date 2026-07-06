"""
caracterizador.py
==================

EL CARACTERIZADOR DE HAZ UNICO -- la misma medicion para RK4 y SIMION.

Por que existe: hasta ahora el gemelo (RK4) y la verdad de terreno
(SIMION) se median con maquinaria distinta, y esa diferencia de
instrumentos contamina la comparacion predicho-vs-real y el dataset de
calibracion de pesos (Task B). Este modulo es la unica fuente de verdad
de COMO se mide un haz: ambos simuladores entran por adaptadores y salen
por la misma funcion `caracterizar()`.

Hace todo lo que el flujo de puntaje necesita:
  - fracciones de resolucion (llegada al plano, hits, perdidas, paredes)
  - distancia al detector por ion y el objetivo actual (best-10% + terminos)
  - features espaciales (offset, sigma, halo, kurtosis)          [Task A]
  - features cinematicas (divergencia, Twiss alfa/beta, emitancia) [C/D]
  - residuo de matriz de transporte (aberraciones)                 [E]
  - combinacion lineal generica de features (enchufe para los pesos
    aprendidos por Ridge/Lasso de la Task B)

Unidades: posiciones mm; velocidades en CUALQUIER unidad consistente
(mm/s del RK4 o mm/usec de SIMION -- las features angulares usan
cocientes vx/vz y son invariantes a la escala); emitancia en mm*mrad.

Adaptadores:
  desde_simion_ultimo_vuelo()   recording completo (posicion+velocidad
                                inicial y final por ion) -> features FULL
  caracterizar(...)             entrada generica (la usan ambos lados)

Los tests viven en playpen/test_caracterizador.py (recuperacion de Twiss
sobre haces sinteticos con parametros conocidos, residuo cero para
transporte perfectamente lineal).
"""

import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

# Ventana del detector (= optimizer.DETECTOR_REGION, duplicada como
# constantes simples para no depender de optimizer.py).
DET_X = (70.0, 82.0)
DET_Y = (70.0, 83.0)
DET_Z = (403.0, 407.0)
DET_CENTER = np.array([(DET_X[0] + DET_X[1]) / 2, (DET_Y[0] + DET_Y[1]) / 2])
_DET_LO = np.array([DET_X[0], DET_Y[0], DET_Z[0]])
_DET_HI = np.array([DET_X[1], DET_Y[1], DET_Z[1]])

Z_PLANO = 390.0          # "llego al plano del detector" = z final > esto
FRACCION_PUNTA = 0.10    # borde de ataque del objetivo (= SPLAT_TOP_FRACTION)
PESO_TRANSMISION = 20.0  # mm  (= HIT_TERM_SCALE_MM)
PESO_PARED = 50.0        # mm  (= WALL_PENALTY_MM, solo aplica lado RK4)


# ----------------------------------------------------------------------
# primitivas compartidas
# ----------------------------------------------------------------------
def distancias_al_detector(posiciones):
    """(N,3) -> (N,) distancia euclidea de cada punto a la caja del
    detector (0 adentro) -- la primitiva del objetivo denso."""
    p = np.asarray(posiciones, dtype=float)
    abajo = np.maximum(_DET_LO - p, 0.0)
    arriba = np.maximum(p - _DET_HI, 0.0)
    return np.linalg.norm(abajo + arriba, axis=1)


def hits_en_ventana(posiciones):
    """Conteo con la MISMA regla estricta de optimizer.count_hits."""
    p = np.asarray(posiciones, dtype=float)
    return int(((p[:, 0] > DET_X[0]) & (p[:, 0] < DET_X[1]) &
                (p[:, 1] > DET_Y[0]) & (p[:, 1] < DET_Y[1]) &
                (p[:, 2] > DET_Z[0]) & (p[:, 2] < DET_Z[1])).sum())


def _twiss_1d(x, xp):
    """Parametros de Twiss de un plano: x (mm), xp = pendiente (rad).
    Devuelve (alfa, beta mm/rad, emitancia_rms mm*mrad)."""
    x = x - x.mean()
    xp = xp - xp.mean()
    s_xx, s_pp, s_xp = (x * x).mean(), (xp * xp).mean(), (x * xp).mean()
    det = s_xx * s_pp - s_xp ** 2
    if det <= 0:
        return np.nan, np.nan, np.nan
    emit = float(np.sqrt(det))            # mm*rad
    return float(-s_xp / emit), float(s_xx / emit), emit * 1e3  # ->mm*mrad


def _kurtosis(x):
    """Exceso de kurtosis (0 = gaussiana)."""
    x = x - x.mean()
    s2 = (x * x).mean()
    return float((x ** 4).mean() / s2 ** 2 - 3.0) if s2 > 0 else np.nan


# ----------------------------------------------------------------------
# LA funcion unica
# ----------------------------------------------------------------------
def caracterizar(pos_final, vel_final=None, pos_inicial=None, vel_inicial=None,
                 mascara=None, flags=None):
    """
    Caracterizacion completa de un haz, identica para RK4 y SIMION.

    Parameters
    ----------
    pos_final : (N,3) posicion final (splat / estado al resolverse), mm
    vel_final : (N,3) velocidad final, opcional (unidad consistente)
    pos_inicial, vel_inicial : (N,3) estado inicial por ion, opcional --
        habilitan el residuo de matriz de transporte (aberraciones)
    mascara : (N,) bool opcional -- que iones considerar (p.ej. las
        particulas "limpias" del RK4); None = todos
    flags : dict opcional con arrays bool del lado RK4
        (reached=..., lost=..., hit_wall=...) -- agrega fracciones y el
        termino de pared al objetivo

    Returns
    -------
    dict de features escalares (NaN donde falta el insumo, nunca error).
    """
    p1 = np.asarray(pos_final, dtype=float)
    n_total = len(p1)
    considerar = np.ones(n_total, dtype=bool) if mascara is None else np.asarray(mascara, bool)

    d = distancias_al_detector(p1[considerar])
    n_keep = max(1, int(np.ceil(considerar.sum() * FRACCION_PUNTA)))
    dist_punta = float(np.sort(d)[:n_keep].mean()) if len(d) else np.nan
    hits = hits_en_ventana(p1[considerar])
    frac_hits = hits / max(1, considerar.sum())

    out = {
        "n_total": int(n_total),
        "n_considerados": int(considerar.sum()),
        "hits": hits,
        "dist_punta_mm": dist_punta,
        "dist_media_mm": float(d.mean()) if len(d) else np.nan,
        "objetivo_mm": dist_punta + PESO_TRANSMISION * (1.0 - frac_hits),
    }

    # fracciones de resolucion (lado RK4 las trae en flags; lado SIMION
    # todo ion termina en un electrodo, asi que solo aplica el plano)
    if flags:
        for nombre, arr in flags.items():
            out[f"{nombre}_fraction"] = float(np.asarray(arr, bool)[considerar].mean())
        if "hit_wall" in flags:
            out["objetivo_mm"] += PESO_PARED * out["hit_wall_fraction"]

    # ------------------------------------------------------- grupo que llega
    llega = considerar & (p1[:, 2] > Z_PLANO)
    out["n_plane"] = int(llega.sum())
    out["plane_fraction"] = float(llega.sum() / max(1, considerar.sum()))

    claves_nan = ["offset_x_mm", "offset_y_mm", "sigma_x_mm", "sigma_y_mm",
                  "halo_fraction", "kurtosis_x", "kurtosis_y",
                  "div_x_mrad", "div_y_mrad",
                  "twiss_alpha_x", "twiss_alpha_y", "twiss_beta_x", "twiss_beta_y",
                  "emittance_x", "emittance_y",
                  "resid_transporte_x_mm", "resid_transporte_y_mm"]
    for k in claves_nan:
        out[k] = float("nan")
    if llega.sum() < 3:
        return out

    g = p1[llega]

    # -- espaciales (Task A: referencia; halo = fuera de 3 sigma radial) --
    out["offset_x_mm"] = float(g[:, 0].mean() - DET_CENTER[0])
    out["offset_y_mm"] = float(g[:, 1].mean() - DET_CENTER[1])
    out["sigma_x_mm"] = float(g[:, 0].std())
    out["sigma_y_mm"] = float(g[:, 1].std())
    r = np.hypot(g[:, 0] - g[:, 0].mean(), g[:, 1] - g[:, 1].mean())
    sr = r.std()
    out["halo_fraction"] = float((r > 3 * sr).mean()) if sr > 0 else 0.0
    out["kurtosis_x"] = _kurtosis(g[:, 0])
    out["kurtosis_y"] = _kurtosis(g[:, 1])

    # -- cinematicas (necesitan velocidad final) --------------------------
    if vel_final is not None:
        v = np.asarray(vel_final, dtype=float)[llega]
        with np.errstate(divide="ignore", invalid="ignore"):
            vz = np.where(np.abs(v[:, 2]) > 1e-30, v[:, 2], np.nan)
            xp = v[:, 0] / vz   # pendientes (rad), invariantes a la unidad
            yp = v[:, 1] / vz
        ok = np.isfinite(xp) & np.isfinite(yp)
        if ok.sum() >= 3:
            out["div_x_mrad"] = float(np.std(xp[ok]) * 1e3)
            out["div_y_mrad"] = float(np.std(yp[ok]) * 1e3)
            a, b, e = _twiss_1d(g[ok, 0], xp[ok])
            out["twiss_alpha_x"], out["twiss_beta_x"], out["emittance_x"] = a, b, e
            a, b, e = _twiss_1d(g[ok, 1], yp[ok])
            out["twiss_alpha_y"], out["twiss_beta_y"], out["emittance_y"] = a, b, e

        # -- residuo de matriz de transporte (Task E) ---------------------
        # mejor mapa LINEAL (x0,y0,x0',y0') -> (x1 / y1) por minimos
        # cuadrados; lo que el mapa no explica es aberracion pura.
        if pos_inicial is not None and vel_inicial is not None and ok.sum() >= 6:
            p0 = np.asarray(pos_inicial, dtype=float)[llega][ok]
            v0 = np.asarray(vel_inicial, dtype=float)[llega][ok]
            with np.errstate(divide="ignore", invalid="ignore"):
                vz0 = np.where(np.abs(v0[:, 2]) > 1e-30, v0[:, 2], np.nan)
                xp0, yp0 = v0[:, 0] / vz0, v0[:, 1] / vz0
            ok0 = np.isfinite(xp0) & np.isfinite(yp0)
            if ok0.sum() >= 6:
                A = np.column_stack([p0[ok0, 0], p0[ok0, 1], xp0[ok0], yp0[ok0],
                                     np.ones(ok0.sum())])
                for eje, clave in ((0, "resid_transporte_x_mm"), (1, "resid_transporte_y_mm")):
                    objetivo_col = g[ok][ok0, eje]
                    coef, *_ = np.linalg.lstsq(A, objetivo_col, rcond=None)
                    res = objetivo_col - A @ coef
                    out[clave] = float(np.sqrt((res ** 2).mean()))
    return out


# ----------------------------------------------------------------------
# adaptadores
# ----------------------------------------------------------------------
def desde_simion_ultimo_vuelo(path=None):
    """Caracterizacion FULL del ultimo vuelo de SIMION (usa el recording
    rico: estado inicial + final por ion -> incluye residuo de transporte)."""
    from beam_characterization import read_last_fly, parse_simion_recording, RECORDING_FILE
    rec = parse_simion_recording(read_last_fly(path or RECORDING_FILE))
    feats = caracterizar(rec["pos1"], rec["vel1"], rec["pos0"], rec["vel0"])
    print(f"[caracterizador] desde_simion_ultimo_vuelo: {feats['n_total']} iones, "
          f"{feats['n_plane']} llegaron al plano, {feats['hits']} en ventana")
    return rec, feats


# ----------------------------------------------------------------------
# OBJETIVO v2 -- normalizado, jerarquia definida el 2026-07-05:
# hits POR ENCIMA DE TODO (0.45 > 0.40 = suma de la escalera de forma),
# luego la escalera: offset > halo > kurtosis > colimacion > twiss.
# El termino de acercamiento (0.15) es el piso de arranque: es el UNICO
# con gradiente cuando ningun ion llega al plano (el ~95% del espacio) y
# se apaga solo (d~0) cuando los terminos de forma cobran sentido.
# Toda feature indefinida (no llego suficiente haz) toma su PEOR valor
# (1.0), nunca NaN: no llegar debe costar el maximo en forma.
# ----------------------------------------------------------------------
# v2.1 (2026-07-05): se agrega el termino de CUERPO (1 - plane_fraction).
# Motivo medido: el objetivo solo miraba la punta del haz y los hits; la
# muerte masiva antes/en el cuadrupolo era invisible. Un barrido de 28
# vuelos guiado por supervivencia del cuerpo encontro quad x0.70 -> 22
# hits (record, 2.2x el mejor de 600+ corridas de optimizacion) y
# regimenes con 75% del haz llegando al plano con 0 hits -- gradiente que
# solo el termino de cuerpo puede explotar. hits sigue dominando
# (0.40 > 0.35 = suma de la escalera de forma).
PESOS_V2 = {
    "transmision": 0.40,
    "cuerpo": 0.12,
    "acercamiento": 0.13,
    "offset": 0.10,
    "halo": 0.08,
    "kurtosis": 0.06,
    "colimacion": 0.06,
    "twiss": 0.05,
}
PESO_PARED_V2 = 0.15   # termino extra SOLO lado RK4 (SIMION no lo puede medir)

D0_MM = 50.0           # escala del acercamiento: d=50mm -> termino 0.5
K0_KURT = 3.0          # kurtosis de referencia (hoy medimos ~6 -> 0.67)
DIV0_MRAD = 100.0      # divergencia de referencia (hoy ~125 -> 0.55)
_MEDIA_VENTANA = np.array([(DET_X[1] - DET_X[0]) / 2, (DET_Y[1] - DET_Y[0]) / 2])


def _racional(x, x0):
    """x/(x+x0): [0,inf) -> [0,1), gradiente maximo cerca de 0."""
    return float(x / (x + x0)) if np.isfinite(x) and x >= 0 else 1.0


def objetivo_v2(features, con_pared=None):
    """
    Objetivo normalizado J en [0,1] (0 = perfecto), minimizar.

    features : dict de caracterizar()
    con_pared: None = automatico (agrega el termino de pared si la
               feature hit_wall_fraction esta presente, i.e. lado RK4)

    Returns
    -------
    (J, desglose) -- desglose trae la contribucion de cada termino ya
    multiplicada por su peso, para ver que domina en cada regimen.
    """
    f = features
    n = max(1, f.get("n_considerados") or f.get("n_total") or 1)
    hits = f.get("hits") or 0

    off_x, off_y = f.get("offset_x_mm", np.nan), f.get("offset_y_mm", np.nan)
    radio_norm = (np.hypot(off_x / _MEDIA_VENTANA[0], off_y / _MEDIA_VENTANA[1])
                  if np.isfinite(off_x) and np.isfinite(off_y) else np.nan)
    alfas = [abs(f.get("twiss_alpha_x", np.nan)), abs(f.get("twiss_alpha_y", np.nan))]
    alfa = np.nanmean(alfas) if np.isfinite(alfas).any() else np.nan
    divs = [f.get("div_x_mrad", np.nan), f.get("div_y_mrad", np.nan)]
    div = np.nanmean(divs) if np.isfinite(divs).any() else np.nan
    kurts = [f.get("kurtosis_x", np.nan), f.get("kurtosis_y", np.nan)]
    kurt = np.nanmax(kurts) if np.isfinite(kurts).any() else np.nan
    halo = f.get("halo_fraction", np.nan)

    plane_fr = f.get("plane_fraction")
    terminos = {
        "transmision": 1.0 - hits / n,
        "cuerpo": 1.0 - float(plane_fr) if plane_fr is not None and np.isfinite(plane_fr) else 1.0,
        "acercamiento": _racional(f.get("dist_punta_mm", np.nan), D0_MM),
        "offset": _racional(radio_norm, 1.0),
        "halo": float(halo) if np.isfinite(halo) else 1.0,
        "kurtosis": _racional(max(kurt, 0.0) if np.isfinite(kurt) else np.nan, K0_KURT),
        "colimacion": _racional(div, DIV0_MRAD),
        "twiss": _racional(alfa ** 2 if np.isfinite(alfa) else np.nan, 1.0),
    }
    desglose = {k: PESOS_V2[k] * v for k, v in terminos.items()}

    if con_pared is None:
        con_pared = "hit_wall_fraction" in f
    if con_pared:
        desglose["pared"] = PESO_PARED_V2 * float(f.get("hit_wall_fraction", 0.0))

    return float(sum(desglose.values())), desglose


def combinacion_lineal(features, pesos):
    """target = sum_i w_i * f_i sobre las claves presentes en `pesos` --
    el enchufe donde entran los pesos aprendidos por la Task B (Ridge/
    Lasso) en lugar de los 3 terminos a mano del objetivo actual."""
    total, usadas = 0.0, 0
    for clave, w in pesos.items():
        val = features.get(clave)
        if val is not None and np.isfinite(val):
            total += w * float(val)
            usadas += 1
    return total, usadas


if __name__ == "__main__":
    rec, feats = desde_simion_ultimo_vuelo()
    print(f"Ultimo vuelo SIMION: {len(rec['ion'])} iones")
    for k, v in feats.items():
        if isinstance(v, (int, np.integer)):
            print(f"  {k:24s} = {v}")
        elif np.isfinite(v):
            print(f"  {k:24s} = {v:.4f}")
