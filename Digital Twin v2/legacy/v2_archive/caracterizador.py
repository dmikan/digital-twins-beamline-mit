"""
v2/caracterizador.py
====================

Unified beam characterization and analysis module.
Combines:
  - caracterizador.py (full beam characterization and objective functions)
  - beam_characterization.py (SIMION recording parser and basic characterization)
  - beam_progress_score.py (RK4 particle trajectory progress and combined scoring)
"""

import pathlib
import re
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RECORDING_FILE = HERE.parent / "out.txt"

# ----------------------------------------------------------------------
# Detector Configuration (constants)
# ----------------------------------------------------------------------
DET_X = (70.0, 82.0)
DET_Y = (70.0, 83.0)
DET_Z = (403.0, 407.0)
DET_CENTER = np.array([(DET_X[0] + DET_X[1]) / 2, (DET_Y[0] + DET_Y[1]) / 2])
_DET_LO = np.array([DET_X[0], DET_Y[0], DET_Z[0]])
_DET_HI = np.array([DET_X[1], DET_Y[1], DET_Z[1]])

Z_PLANO = 390.0          # "reached detector plane" z-threshold
FRACCION_PUNTA = 0.10    # top fraction of particles for closest approach
PESO_TRANSMISION = 20.0  # mm scale for transmission penalty
PESO_PARED = 50.0        # mm scale for wall penalty (RK4 only)

EMIT0_KIN = 500.0        # emittance reference (mm*mrad)
KIN_MIN_PARTICULAS = 5   # min particles crossing plane to evaluate kinematics

PESOS_V2 = {
    "transmision": 0.50,
    "cuerpo": 0.10,
    "acercamiento": 0.13,
    "offset": 0.08,
    "halo": 0.06,
    "kurtosis": 0.05,
    "colimacion": 0.04,
    "twiss": 0.04,
}
PESO_PARED_V2 = 0.15

H0_HITS = 15.0
D0_MM = 50.0
K0_KURT = 3.0
DIV0_MRAD = 100.0
_MEDIA_VENTANA = np.array([(DET_X[1] - DET_X[0]) / 2, (DET_Y[1] - DET_Y[0]) / 2])

# ----------------------------------------------------------------------
# Basic Math and Utilities
# ----------------------------------------------------------------------
def distancias_al_detector(posiciones):
    p = np.asarray(posiciones, dtype=float)
    abajo = np.maximum(_DET_LO - p, 0.0)
    arriba = np.maximum(p - _DET_HI, 0.0)
    return np.linalg.norm(abajo + arriba, axis=1)

def hits_en_ventana(posiciones):
    p = np.asarray(posiciones, dtype=float)
    return int(((p[:, 0] > DET_X[0]) & (p[:, 0] < DET_X[1]) &
                (p[:, 1] > DET_Y[0]) & (p[:, 1] < DET_Y[1]) &
                (p[:, 2] > DET_Z[0]) & (p[:, 2] < DET_Z[1])).sum())

def _twiss_1d(x, xp):
    x = x - x.mean()
    xp = xp - xp.mean()
    s_xx, s_pp, s_xp = (x * x).mean(), (xp * xp).mean(), (x * xp).mean()
    det = s_xx * s_pp - s_xp ** 2
    if det <= 0:
        return np.nan, np.nan, np.nan
    emit = float(np.sqrt(det))
    return float(-s_xp / emit), float(s_xx / emit), emit * 1e3

def _kurtosis(x):
    x = x - x.mean()
    s2 = (x * x).mean()
    return float((x ** 4).mean() / s2 ** 2 - 3.0) if s2 > 0 else np.nan

def _racional(x, x0):
    return float(x / (x + x0)) if np.isfinite(x) and x >= 0 else 1.0

# ----------------------------------------------------------------------
# Full Beam Characterization
# ----------------------------------------------------------------------
def caracterizar(pos_final, vel_final=None, pos_inicial=None, vel_inicial=None,
                  mascara=None, flags=None):
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

    if flags:
        for nombre, arr in flags.items():
            out[f"{nombre}_fraction"] = float(np.asarray(arr, bool)[considerar].mean())
        if "hit_wall" in flags:
            out["objetivo_mm"] += PESO_PARED * out["hit_wall_fraction"]

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
    out["offset_x_mm"] = float(g[:, 0].mean() - DET_CENTER[0])
    out["offset_y_mm"] = float(g[:, 1].mean() - DET_CENTER[1])
    out["sigma_x_mm"] = float(g[:, 0].std())
    out["sigma_y_mm"] = float(g[:, 1].std())
    r = np.hypot(g[:, 0] - g[:, 0].mean(), g[:, 1] - g[:, 1].mean())
    sr = r.std()
    out["halo_fraction"] = float((r > 3 * sr).mean()) if sr > 0 else 0.0
    out["kurtosis_x"] = _kurtosis(g[:, 0])
    out["kurtosis_y"] = _kurtosis(g[:, 1])

    if vel_final is not None:
        v = np.asarray(vel_final, dtype=float)[llega]
        with np.errstate(divide="ignore", invalid="ignore"):
            vz = np.where(np.abs(v[:, 2]) > 1e-30, v[:, 2], np.nan)
            xp = v[:, 0] / vz
            yp = v[:, 1] / vz
        ok = np.isfinite(xp) & np.isfinite(yp)
        if ok.sum() >= 3:
            out["div_x_mrad"] = float(np.std(xp[ok]) * 1e3)
            out["div_y_mrad"] = float(np.std(yp[ok]) * 1e3)
            a, b, e = _twiss_1d(g[ok, 0], xp[ok])
            out["twiss_alpha_x"], out["twiss_beta_x"], out["emittance_x"] = a, b, e
            a, b, e = _twiss_1d(g[ok, 1], yp[ok])
            out["twiss_alpha_y"], out["twiss_beta_y"], out["emittance_y"] = a, b, e

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
# Objective v2 Functions
# ----------------------------------------------------------------------
def objetivo_v2(features, con_pared=None):
    f = features
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
        "transmision": 1.0 / (1.0 + hits / H0_HITS),
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
    total, usadas = 0.0, 0
    for clave, w in pesos.items():
        val = features.get(clave)
        if val is not None and np.isfinite(val):
            total += w * float(val)
            usadas += 1
    return total, usadas

# ----------------------------------------------------------------------
# SIMION Recording Parsing
# ----------------------------------------------------------------------
_NUM = r"([-+0-9.eE]+)"

def read_last_fly(path=RECORDING_FILE):
    text = pathlib.Path(path).read_text(errors="replace")
    blocks = text.split("Begin Fly'm")
    if len(blocks) < 2:
        raise ValueError(f"{path}: no se encontro ningun bloque de vuelo")
    return blocks[-1]

def parse_simion_recording(fly_text):
    records = [" ".join(r.split()) for r in re.split(r"\n\s*\n", fly_text) if "Event(" in r]

    def field(rec, name):
        m = re.search(re.escape(name) + r"\(" + _NUM, rec)
        return float(m.group(1)) if m else np.nan

    first, last = {}, {}
    for rec in records:
        m = re.search(r"Ion\((\d+)\) Event\(([^)]+)\)", rec)
        if not m:
            continue
        ion, event = int(m.group(1)), m.group(2)
        entry = dict(
            event=event,
            pos=[field(rec, "X"), field(rec, "Y"), field(rec, "Z")],
            vel=[field(rec, "Vx"), field(rec, "Vy"), field(rec, "Vz")],
            ke=field(rec, "KE"), tof=field(rec, "TOF"),
        )
        if event == "Ion Created":
            first[ion] = entry
        else:
            last[ion] = entry

    ions = sorted(set(first) & set(last))
    return dict(
        ion=np.array(ions),
        pos0=np.array([first[i]["pos"] for i in ions]),
        vel0=np.array([first[i]["vel"] for i in ions]),
        ke0=np.array([first[i]["ke"] for i in ions]),
        pos1=np.array([last[i]["pos"] for i in ions]),
        vel1=np.array([last[i]["vel"] for i in ions]),
        ke1=np.array([last[i]["ke"] for i in ions]),
        tof=np.array([last[i]["tof"] for i in ions]),
        event=np.array([last[i]["event"] for i in ions]),
    )

def desde_simion_ultimo_vuelo(path=None):
    rec = parse_simion_recording(read_last_fly(path or RECORDING_FILE))
    feats = caracterizar(rec["pos1"], rec["vel1"], rec["pos0"], rec["vel0"])
    print(f"[caracterizador] desde_simion_ultimo_vuelo: {feats['n_total']} iones, "
          f"{feats['n_plane']} al plano, {feats['hits']} hits")
    return rec, feats

def characterize_beam(positions, velocities=None):
    positions = np.asarray(positions, dtype=float)
    near = positions[:, 2] > Z_PLANO
    p = positions[near]
    v = None if velocities is None else np.asarray(velocities, dtype=float)[near]

    out = {
        "n_total": int(len(positions)),
        "n_plane": int(len(p)),
        "plane_fraction": float(len(p) / max(1, len(positions))),
    }
    nan_keys = [
        "offset_x_mm", "offset_y_mm", "sigma_x_mm", "sigma_y_mm",
        "halo_fraction", "kurtosis_x", "kurtosis_y",
        "div_x_mrad", "div_y_mrad",
        "twiss_alpha_x", "twiss_alpha_y",
        "twiss_beta_x", "twiss_beta_y",
        "emittance_x", "emittance_y",
    ]
    for k in nan_keys:
        out[k] = float("nan")

    if len(p) >= 3:
        out["offset_x_mm"] = float(p[:, 0].mean() - DET_CENTER[0])
        out["offset_y_mm"] = float(p[:, 1].mean() - DET_CENTER[1])
        out["sigma_x_mm"] = float(p[:, 0].std())
        out["sigma_y_mm"] = float(p[:, 1].std())

    if v is not None and len(p) >= 3:
        with np.errstate(divide="ignore", invalid="ignore"):
            vz = np.where(np.abs(v[:, 2]) > 1e-12, v[:, 2], np.nan)
            out["div_x_mrad"] = float(np.nanstd(v[:, 0] / vz) * 1e3)
            out["div_y_mrad"] = float(np.nanstd(v[:, 1] / vz) * 1e3)

    return out

def characterize_last_fly(path=RECORDING_FILE):
    rec = parse_simion_recording(read_last_fly(path))
    feats = characterize_beam(rec["pos1"], rec["vel1"])
    return rec, feats

# ----------------------------------------------------------------------
# Beam Generator
# ----------------------------------------------------------------------
def make_beam(N, species, start_point, mean_energy_eV, std_energy_eV,
              half_angle_deg, cone_axis=(-1.0, 0.0, 0.0), seed=None):
    rng = np.random.default_rng(seed)
    e = 1.602176634e-19

    start_positions = np.zeros((N, 3))
    start_positions[:] = start_point

    energies = rng.normal(mean_energy_eV, std_energy_eV, N)
    energies = np.clip(energies, 0, None)
    energies *= e

    half_angle = np.deg2rad(half_angle_deg)
    cos_theta = rng.uniform(np.cos(half_angle), 1.0, N)
    theta = np.arccos(cos_theta)
    phi = rng.uniform(0, 2 * np.pi, N)

    axis = np.asarray(cone_axis, dtype=float)
    axis /= np.linalg.norm(axis)

    directions = np.empty((N, 3))
    directions[:, 0] = axis[0] * np.cos(theta)
    directions[:, 1] = np.sin(theta) * np.cos(phi)
    directions[:, 2] = np.sin(theta) * np.sin(phi)
    directions /= np.linalg.norm(directions, axis=1)[:, None]

    speeds = 1000 * np.sqrt(2 * energies / species.mass)
    start_velocities = speeds[:, None] * directions

    return start_positions, start_velocities

# ----------------------------------------------------------------------
# Kinematics in Intermediate Planes
# ----------------------------------------------------------------------
def cinematica_en_plano(posiciones, velocidades, stop_idx=None, paso_pared=None,
                        z_plano=Z_PLANO, eje=2, direccion=1):
    p = np.asarray(posiciones, dtype=float)
    v = np.asarray(velocidades, dtype=float)
    T, N = p.shape[0], p.shape[1]
    t1, t2 = [a for a in (0, 1, 2) if a != eje]
    coord = p[:, :, eje]

    pasos = np.arange(T)[:, None]
    tope = np.asarray(stop_idx)[None, :] if stop_idx is not None else T - 1
    cruzo = (direccion * (coord - z_plano) >= 0) & (pasos <= tope)
    hay = cruzo.any(axis=0)
    k = np.argmax(cruzo, axis=0)
    valida = hay & (k > 0)
    if paso_pared is not None:
        pared = np.asarray(paso_pared)
        valida &= (pared < 0) | (k <= pared)

    out = {"n_cruzan": int(valida.sum()),
           "centro_x_mm": np.nan, "centro_y_mm": np.nan,
           "sigma_x_mm": np.nan, "sigma_y_mm": np.nan,
           "div_x_mrad": np.nan, "div_y_mrad": np.nan,
           "twiss_alpha_x": np.nan, "twiss_alpha_y": np.nan,
           "emittance_x": np.nan, "emittance_y": np.nan}
    if valida.sum() < 3:
        return out

    idx = np.where(valida, k, 1)
    cols = np.arange(N)
    p0, p1 = p[idx - 1, cols], p[idx, cols]
    v0, v1 = v[idx - 1, cols], v[idx, cols]
    dc = p1[:, eje] - p0[:, eje]
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.clip(np.where(np.abs(dc) > 1e-30, (z_plano - p0[:, eje]) / dc, 0.0), 0.0, 1.0)
    pc = (p0 + f[:, None] * (p1 - p0))[valida]
    vc = (v0 + f[:, None] * (v1 - v0))[valida]

    out["centro_x_mm"] = float(pc[:, t1].mean())
    out["centro_y_mm"] = float(pc[:, t2].mean())
    out["sigma_x_mm"] = float(pc[:, t1].std())
    out["sigma_y_mm"] = float(pc[:, t2].std())

    with np.errstate(divide="ignore", invalid="ignore"):
        v_l = np.where(np.abs(vc[:, eje]) > 1e-30, vc[:, eje], np.nan)
        xp, yp = vc[:, t1] / v_l, vc[:, t2] / v_l
    ok = np.isfinite(xp) & np.isfinite(yp)
    out["n_cruzan"] = int(ok.sum())
    if ok.sum() < 3:
        return out
    out["div_x_mrad"] = float(np.std(xp[ok]) * 1e3)
    out["div_y_mrad"] = float(np.std(yp[ok]) * 1e3)
    a, _, e = _twiss_1d(pc[ok, t1], xp[ok])
    out["twiss_alpha_x"], out["emittance_x"] = a, e
    a, _, e = _twiss_1d(pc[ok, t2], yp[ok])
    out["twiss_alpha_y"], out["emittance_y"] = a, e
    return out

def puntaje_cinematico(feats, min_particulas=KIN_MIN_PARTICULAS):
    if (feats.get("n_cruzan") or 0) < min_particulas:
        return 1.0
    divs = [feats.get("div_x_mrad", np.nan), feats.get("div_y_mrad", np.nan)]
    div = np.nanmean(divs) if np.isfinite(divs).any() else np.nan
    alfas = [abs(feats.get("twiss_alpha_x", np.nan)), abs(feats.get("twiss_alpha_y", np.nan))]
    alfa = np.nanmean(alfas) if np.isfinite(alfas).any() else np.nan
    emits = [feats.get("emittance_x", np.nan), feats.get("emittance_y", np.nan)]
    emit = np.nanmean(emits) if np.isfinite(emits).any() else np.nan
    t_col = _racional(div, DIV0_MRAD)
    t_twi = _racional(alfa ** 2 if np.isfinite(alfa) else np.nan, 1.0)
    t_emi = _racional(emit, EMIT0_KIN)
    return float((t_col + t_twi + t_emi) / 3.0)

# ----------------------------------------------------------------------
# Trajectory Scorer
# ----------------------------------------------------------------------
class BeamProgressScorer:
    def __init__(self, bfm, Trajectory, dt, num_steps,
                 detector_bbox=None, travel_axis=0, travel_direction=-1,
                 target=None, bbox=None, wall_index=None, wall_hit_margin=1.0,
                 wall_check_midpoints=True, wall_check_stride=1,
                 terminate_on_wall_hit=False):
        self.bfm = bfm
        self._Trajectory = Trajectory
        self.start_positions = self._Trajectory.initial_state.position
        self.start_velocities = self._Trajectory.initial_state.velocity
        self.n_particles = self.start_positions.shape[0]
        self.dt = dt
        self.num_steps = num_steps
        self.travel_axis = travel_axis
        self.travel_direction = travel_direction
        self.wall_index = wall_index
        self.wall_hit_margin = wall_hit_margin
        self.wall_check_midpoints = wall_check_midpoints
        self.wall_check_stride = max(1, int(wall_check_stride))
        self.terminate_on_wall_hit = terminate_on_wall_hit

        if bbox is None:
            bbox = (bfm.x.min(), bfm.x.max(),
                    bfm.y.min(), bfm.y.max(),
                    bfm.z.min(), bfm.z.max())
        self.bbox = bbox

        lo = [bbox[0], bbox[2], bbox[4]]
        hi = [bbox[1], bbox[3], bbox[5]]
        self._lo, self._hi = lo, hi

        self.detector_bbox = detector_bbox
        if detector_bbox is not None:
            self._det_lo = [detector_bbox[0], detector_bbox[2], detector_bbox[4]]
            self._det_hi = [detector_bbox[1], detector_bbox[3], detector_bbox[5]]
        else:
            self._det_lo = self._det_hi = None

        if target is None:
            if detector_bbox is not None:
                target = self._det_hi[travel_axis] if travel_direction < 0 else self._det_lo[travel_axis]
            else:
                target = lo[travel_axis] if travel_direction < 0 else hi[travel_axis]
        self.target = target
        self.start_coord = self.start_positions[:, travel_axis]

    def score(self, voltages):
        states = np.asarray(self._Trajectory.states)
        positions = np.array([s.position for s in states])
        coord = positions[:, :, self.travel_axis]

        if self.detector_bbox is not None:
            reached_mask = np.ones(coord.shape, dtype=bool)
            for a in (0, 1, 2):
                c = positions[:, :, a]
                reached_mask &= (c >= self._det_lo[a]) & (c <= self._det_hi[a])
        else:
            if self.travel_direction < 0:
                reached_mask = coord <= self.target
            else:
                reached_mask = coord >= self.target

        lost_mask = np.zeros(coord.shape, dtype=bool)
        for a in (0, 1, 2):
            if a == self.travel_axis:
                continue
            c = positions[:, :, a]
            lost_mask |= (c < self._lo[a]) | (c > self._hi[a])

        if self.travel_direction < 0:
            lost_mask |= coord > self._hi[self.travel_axis]
        else:
            lost_mask |= coord < self._lo[self.travel_axis]

        T, N = coord.shape
        stopped = reached_mask | lost_mask
        never_stopped = ~stopped.any(axis=0)
        stop_idx = np.where(never_stopped, T - 1, np.argmax(stopped, axis=0))

        hit_wall = np.zeros(N, dtype=bool)
        hit_wall_step = np.full(N, -1, dtype=int)
        if self.wall_index is not None:
            hit_wall_mask = self._wall_hit_mask(positions, stop_idx)
            any_hit = hit_wall_mask.any(axis=0)
            hit_wall = any_hit
            hit_wall_step = np.where(any_hit, np.argmax(hit_wall_mask, axis=0), -1)

            if self.terminate_on_wall_hit and any_hit.any():
                steps_col = np.arange(T)[:, None]
                dead = hit_wall[None, :] & (steps_col >= hit_wall_step[None, :])
                reached_mask &= ~dead
                lost_mask &= ~dead
                stopped = reached_mask | lost_mask
                never_stopped = ~stopped.any(axis=0)
                stop_idx = np.where(never_stopped, T - 1, np.argmax(stopped, axis=0))
                stop_idx = np.where(hit_wall, np.minimum(stop_idx, hit_wall_step), stop_idx)

        progress = np.zeros(N)
        reached_target = np.zeros(N, dtype=bool)
        lost = np.zeros(N, dtype=bool)
        lost_step = np.full(N, -1, dtype=int)
        span = self.target - self.start_coord

        for p in range(N):
            reached_idx = int(np.argmax(reached_mask[:, p])) if reached_mask[:, p].any() else None
            lost_idx = int(np.argmax(lost_mask[:, p])) if lost_mask[:, p].any() else None

            if reached_idx is not None and (lost_idx is None or reached_idx <= lost_idx):
                progress[p] = 1.0
                reached_target[p] = True
            elif lost_idx is not None:
                lost[p] = True
                lost_step[p] = lost_idx
                frac = (coord[lost_idx, p] - self.start_coord[p]) / span[p] if span[p] != 0 else 0.0
                progress[p] = np.clip(frac, 0.0, 1.0)
            else:
                frac = (coord[-1, p] - self.start_coord[p]) / span[p] if span[p] != 0 else 0.0
                progress[p] = np.clip(frac, 0.0, 1.0)

        result = {
            "mean_progress": float(progress.mean()),
            "survival_rate": float(reached_target.mean()),
            "progress": progress,
            "reached_target": reached_target,
            "lost": lost,
            "lost_step": lost_step,
            "voltages": np.asarray(voltages, dtype=float),
        }
        if self.wall_index is not None:
            result["hit_wall"] = hit_wall
            result["hit_wall_step"] = hit_wall_step
            result["wall_hit_fraction"] = float(hit_wall.mean())

        result["positions"] = positions
        result["stop_idx"] = stop_idx
        return result

    def _wall_hit_mask(self, positions, stop_idx):
        T, N = positions.shape[0], positions.shape[1]
        step_idx = np.arange(T)[:, None]
        active_mask = step_idx <= stop_idx[None, :]

        if self.wall_check_stride > 1:
            stride_mask = (step_idx % self.wall_check_stride == 0)
            final_step_mask = (step_idx == stop_idx[None, :])
            keep_mask = active_mask & (stride_mask | final_step_mask)
        else:
            keep_mask = active_mask

        hit_wall_mask = np.zeros((T, N), dtype=bool)
        tt, pp = np.nonzero(keep_mask)
        pts = positions[tt, pp]
        if len(pts) > 0:
            dist = self.wall_index.distance(pts)
            hit = dist <= self.wall_hit_margin
            hit_wall_mask[tt[hit], pp[hit]] = True

        if T > 1 and self.wall_check_midpoints:
            mids = 0.5 * (positions[:-1] + positions[1:])
            arrival_kept = keep_mask[1:]
            mt, mp = np.nonzero(arrival_kept)
            mpts = mids[mt, mp]
            if len(mpts) > 0:
                mdist = self.wall_index.distance(mpts)
                mhit = mdist <= self.wall_hit_margin
                hit_wall_mask[mt[mhit] + 1, mp[mhit]] = True

        return hit_wall_mask

    def _target_distance(self, positions, stop_idx):
        T, N = positions.shape[0], positions.shape[1]
        det_lo = np.asarray(self._det_lo)
        det_hi = np.asarray(self._det_hi)

        below = np.maximum(det_lo - positions, 0.0)
        below += np.maximum(positions - det_hi, 0.0)
        dist = np.linalg.norm(below, axis=2)

        step_idx = np.arange(T)[:, None]
        active_mask = step_idx <= stop_idx[None, :]
        dist = np.where(active_mask, dist, np.inf)
        return dist.min(axis=0)

    def combined_score(self, voltages, target_weight=1.0, wall_weight=1.0,
                        lost_weight=0.3, target_scale=150.0):
        if self.detector_bbox is None:
            raise ValueError("combined_score needs detector_bbox set")
        if self.wall_index is None:
            raise ValueError("combined_score needs wall_index set")

        result = self.score(voltages)
        positions = result["positions"]
        stop_idx = result["stop_idx"]

        target_distance = self._target_distance(positions, stop_idx)
        target_reward = np.exp(-target_distance / target_scale)

        lost_fraction = float(result["lost"].mean())
        combined = (
            target_weight * float(target_reward.mean())
            - wall_weight * result["wall_hit_fraction"]
            - lost_weight * lost_fraction
        )

        result["target_distance"] = target_distance
        result["target_reward"] = target_reward
        result["combined_score"] = combined
        return result

    def __call__(self, voltages):
        return self.score(voltages)
