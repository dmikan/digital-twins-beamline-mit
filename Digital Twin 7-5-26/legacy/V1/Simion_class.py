import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Event:
    """
    Stores the information corresponding to one SIMION event.
    Example:
        - Ion Created
        - Hit Electrode
    """

    name: str

    TOF: float
    Mass: float
    Charge: float

    X: float
    Y: float
    Z: float

    Vt: float
    Azm: float
    Elv: float

    KE_Error: float

    def as_array(self):
        """Return the event as a NumPy array."""
        return np.array([
            self.TOF,
            self.Mass,
            self.Charge,
            self.X,
            self.Y,
            self.Z,
            self.Vt,
            self.Azm,
            self.Elv,
            self.KE_Error
        ], dtype=float)


# -------------------- ION --------------------

@dataclass
class Ion:
    """
    Represents one ion and all of its recorded events.
    """

    number: int
    events: dict[str, Event] = field(default_factory=dict)

    def add_event(self, event: Event):
        self.events[event.name] = event

    def __getitem__(self, event_name: str):
        return self.events[event_name]


# -------------------- SIMION RUN --------------------

@dataclass
class SimionRun:
    """
    Contains the complete SIMION simulation.
    """

    filename: str | Path

    ions: list[Ion] = field(default_factory=list)

    columns: list[str] = field(default_factory=lambda: [
        "TOF",
        "Mass",
        "Charge",
        "X",
        "Y",
        "Z",
        "Vt",
        "Azm",
        "Elv",
        "KE_Error"
    ])

    event_names: list[str] = field(default_factory=list)

    data: dict[str, np.ndarray] = field(default_factory=dict)

    def add_ion(self, ion: Ion):
        self.ions.append(ion)

    @property
    def nions(self):
        return len(self.ions)

    def __getitem__(self, ion_number: int):
        """
        Access an ion by its number.

        Example:
            run[15]
        """
        return self.ions[ion_number - 1]

# ---------------- Create a fake SIMION run ----------------

run = SimionRun("out.txt")

event_names = ["Ion Created", "Hit Electrode"]

for ion_number in range(1, 6):

    ion = Ion(number=ion_number)

    for event_name in event_names:

        event = Event(
            name=event_name,

            TOF=np.random.uniform(0, 100),
            Mass=np.random.choice([1, 2, 4, 28, 40]),
            Charge=np.random.choice([1, 2]),

            X=np.random.uniform(-100, 100),
            Y=np.random.uniform(-100, 100),
            Z=np.random.uniform(-100, 100),

            Vt=np.random.uniform(0, 10),
            Azm=np.random.uniform(-180, 180),
            Elv=np.random.uniform(-90, 90),

            KE_Error=np.random.uniform(0, 1e-6)
        )

        ion.add_event(event)

    run.add_ion(ion)

run.event_names = event_names

# ---------------- Test the classes ----------------

print("=" * 50)
print("Simulation Information")
print("=" * 50)

print(f"Filename : {run.filename}")
print(f"Number of ions : {run.nions}")
print(f"Events : {run.event_names}")
print(f"Columns : {run.columns}")

print()

# --------------------------------------------------

print("=" * 50)
print("Ion 3")
print("=" * 50)

ion = run[3]

print("Ion number:", ion.number)

print()

created = ion["Ion Created"]

print("Created Position")
print(created.X, created.Y, created.Z)

print("Created TOF")
print(created.TOF)

print()

hit = ion["Hit Electrode"]

print("Impact Position")
print(hit.X, hit.Y, hit.Z)

print("Impact TOF")
print(hit.TOF)

print()

print("=" * 50)
print("Numpy array representation")
print("=" * 50)

print(hit.as_array())

print()

print("=" * 50)
print("Iterate over all ions")
print("=" * 50)

for ion in run.ions:

    hit = ion["Hit Electrode"]

    print(
        f"Ion {ion.number:2d} "
        f"X={hit.X:8.2f} "
        f"Y={hit.Y:8.2f} "
        f"Z={hit.Z:8.2f} "
        f"TOF={hit.TOF:8.3f}"
    )