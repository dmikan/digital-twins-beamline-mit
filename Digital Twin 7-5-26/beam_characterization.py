"""
beam_characterization.py
=========================

Interfaz compartida del handoff "Beam Characterization Features" + la
extraccion de estado completo (posicion + VELOCIDAD + energia) desde el
recording de SIMION.

DESCUBRIMIENTO QUE SIMPLIFICA TODO: SIMION ya registra las velocidades.
El archivo out.txt (--recording-output, escrito junto al .iob) contiene,
por cada ion, un evento "Ion Created" (estado inicial) y un evento
terminal "Hit Electrode" (estado en el splat), cada uno con X,Y,Z,
Vx,Vy,Vz (mm/usec), KE (eV), TOF (usec). optimizer.get_positions() solo
parseaba el texto de consola y descartaba el resto. Aca se parsea el
recording completo. El archivo ACUMULA vuelos (un bloque "Begin Fly'm"
por corrida, ~500 iones c/u), asi que siempre se toma el ULTIMO bloque.

Unidades: posiciones en mm; velocidades del recording en mm/usec
(= 1e6 mm/s = km/s); KE en eV; TOF en usec.

INTERFAZ COMPARTIDA (contrato del handoff -- construir contra esto):

    characterize_beam(positions, velocities=None) -> dict
      positions : (N, 3) posiciones de splat
      velocities: (N, 3) o None -- si una feature necesita velocidad y
                  recibe None, devuelve NaN para esa clave (no error).
      Devuelve un dict de features escalares con nombre.

Reparto (ver handoff): Task A (features espaciales) las llena Partner A;
Task C/D (Twiss/emitancia) Partner C -- las claves ya estan reservadas y
devuelven NaN hasta que se implementen. Este modulo aporta la extraccion
(mia) y features de referencia minimas para poder probar la tuberia
end-to-end.

Demo / chequeo de aceptacion:
    python beam_characterization.py     (parsea el ultimo vuelo de out.txt)
"""

import pathlib
import re

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RECORDING_FILE = HERE / "out.txt"

MM_PER_USEC_TO_MM_PER_S = 1.0e6  # mm/usec -> mm/s

# Ventana del detector (= optimizer.DETECTOR_REGION; duplicada aqui como
# constantes simples para que el modulo no dependa de optimizer.py).
DET_X = (70.0, 82.0)
DET_Y = (70.0, 83.0)
DET_Z = (403.0, 407.0)
DET_CENTER = np.array([(DET_X[0] + DET_X[1]) / 2, (DET_Y[0] + DET_Y[1]) / 2])

_NUM = r"([-+0-9.eE]+)"


def read_last_fly(path=RECORDING_FILE):
    """Texto del ULTIMO bloque de vuelo del recording (el archivo acumula
    todos los vuelos historicos, ~500 iones por bloque)."""
    text = pathlib.Path(path).read_text(errors="replace")
    blocks = text.split("Begin Fly'm")
    if len(blocks) < 2:
        raise ValueError(f"{path}: no se encontro ningun bloque de vuelo")
    return blocks[-1]


def parse_simion_recording(fly_text):
    """
    Extrae el estado inicial y final de cada ion de un bloque de vuelo.

    Returns
    -------
    dict con arrays alineados por ion (orden de aparicion):
      ion        (N,)   numero de ion en SIMION
      pos0, pos1 (N,3)  posicion inicial / de splat, mm
      vel0, vel1 (N,3)  velocidad inicial / en el splat, mm/usec
      ke0, ke1   (N,)   energia cinetica inicial / final, eV
      tof        (N,)   tiempo de vuelo hasta el splat, usec
      event      (N,)   tipo de evento terminal (str)
    """
    # Los registros estan separados por lineas en blanco; los campos de un
    # registro pueden venir partidos en varias lineas -> unir cada registro.
    records = [" ".join(r.split()) for r in re.split(r"\n\s*\n", fly_text)
               if "Event(" in r]

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
            last[ion] = entry  # el ultimo evento no-creacion gana

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


def characterize_beam(positions, velocities=None):
    """
    Contrato compartido del handoff: features escalares con nombre a partir
    de los splats (y velocidades si estan disponibles).

    Las features se calculan sobre el GRUPO QUE LLEGA al plano del detector
    (z > 390 mm) -- medimos que ese subgrupo es el que discrimina, no el
    haz completo que muere aguas arriba.
    """
    positions = np.asarray(positions, dtype=float)
    near = positions[:, 2] > 390.0
    p = positions[near]
    v = None if velocities is None else np.asarray(velocities, dtype=float)[near]

    out = {
        "n_total": int(len(positions)),
        "n_plane": int(len(p)),
        "plane_fraction": float(len(p) / max(1, len(positions))),
    }
    nan_keys = [
        "offset_x_mm", "offset_y_mm", "sigma_x_mm", "sigma_y_mm",
        "halo_fraction", "kurtosis_x", "kurtosis_y",          # Task A (Partner A)
        "div_x_mrad", "div_y_mrad",                            # necesita velocities
        "twiss_alpha_x", "twiss_alpha_y",                      # Task C/D (Partner C)
        "twiss_beta_x", "twiss_beta_y",
        "emittance_x", "emittance_y",
    ]
    for k in nan_keys:
        out[k] = float("nan")

    if len(p) >= 3:
        # referencia minima para poder probar la tuberia end-to-end;
        # Partner A es dueno de extender/reemplazar (halo, kurtosis).
        out["offset_x_mm"] = float(p[:, 0].mean() - DET_CENTER[0])
        out["offset_y_mm"] = float(p[:, 1].mean() - DET_CENTER[1])
        out["sigma_x_mm"] = float(p[:, 0].std())
        out["sigma_y_mm"] = float(p[:, 1].std())

    if v is not None and len(p) >= 3:
        # divergencia: angulo transversal respecto del eje de viaje (+z)
        with np.errstate(divide="ignore", invalid="ignore"):
            vz = np.where(np.abs(v[:, 2]) > 1e-12, v[:, 2], np.nan)
            out["div_x_mrad"] = float(np.nanstd(v[:, 0] / vz) * 1e3)
            out["div_y_mrad"] = float(np.nanstd(v[:, 1] / vz) * 1e3)

    return out


def characterize_last_fly(path=RECORDING_FILE):
    """Conveniencia: parsea el ultimo vuelo y lo caracteriza."""
    rec = parse_simion_recording(read_last_fly(path))
    feats = characterize_beam(rec["pos1"], rec["vel1"])
    return rec, feats


if __name__ == "__main__":
    rec, feats = characterize_last_fly()
    n = len(rec["ion"])
    print(f"Ultimo vuelo: {n} iones parseados")
    print(f"eventos terminales: {dict(zip(*np.unique(rec['event'], return_counts=True)))}")
    print(f"KE inicial: {rec['ke0'].mean():.1f} eV (media)   "
          f"KE final: {rec['ke1'].mean():.1f} eV (media)")
    inside = ((rec["pos1"][:, 0] > DET_X[0]) & (rec["pos1"][:, 0] < DET_X[1]) &
              (rec["pos1"][:, 1] > DET_Y[0]) & (rec["pos1"][:, 1] < DET_Y[1]) &
              (rec["pos1"][:, 2] > DET_Z[0]) & (rec["pos1"][:, 2] < DET_Z[1]))
    print(f"hits (ventana del detector): {inside.sum()}")
    print("\ncharacterize_beam(pos, vel):")
    for k, v in feats.items():
        print(f"  {k:16s} = {v}")
