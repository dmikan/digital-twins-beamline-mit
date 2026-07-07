"""Simple tests for BeamProgressScorer using mock integrator/trajectory.

Run with:
    python beam_progress_scorer_test.py

The tests are lightweight and do not require SIMION or real RK4 code.
"""
from types import SimpleNamespace
import numpy as np

from beam_progress_score import make_beam, BeamProgressScorer


class MockBFM:
    def __init__(self, x, y, z):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.z = np.asarray(z, dtype=float)
        # placeholder potential/field
        self.V = None

    def set_voltages(self, voltages):
        # simple store; real code might recompute V/E
        self.V = np.asarray(voltages, dtype=float)


class MockBeam:
    def __init__(self, species, start_positions, start_velocities):
        # keep arrays as provided
        self.species = species
        self.position = np.asarray(start_positions, dtype=float)
        self.velocity = np.asarray(start_velocities, dtype=float)
        self.n_particles = self.position.shape[0]


class MockState:
    def __init__(self, position):
        self.position = np.asarray(position, dtype=float)


class MockTrajectory:
    def __init__(self, beam: MockBeam):
        # states will be a list of snapshot objects with .position (N,3)
        self.states = [MockState(beam.position.copy())]


class MockRK4Integrator:
    def __init__(self, bfm):
        self.bfm = bfm

    def integrate(self, trajectory: MockTrajectory, dt: float, num_steps: int):
        # simple straight-line ballistic propagation using trajectory.states[0]
        pos0 = trajectory.states[0].position  # (N,3)
        # For tests we assume velocities are stored in the beam (we can't access beam here),
        # so instead we will embed velocities into the initial state object if present.
        # To keep the contract simple, we expect the initial state's position attribute
        # to be a tuple (positions, velocities) if velocities are needed.
        # But above MockTrajectory was created from MockBeam, so instead we attach
        # velocities to the trajectory object for the test setup.
        if not hasattr(trajectory, "velocities"):
            raise RuntimeError("MockTrajectory requires .velocities attribute set by test setup")

        velocities = trajectory.velocities  # (N,3) mm/s
        # produce num_steps+1 snapshots (including initial)
        for step in range(1, num_steps + 1):
            t = step * dt
            newpos = pos0 + velocities * t
            trajectory.states.append(MockState(newpos))


# Simple species mock
class Species:
    def __init__(self, mass):
        self.mass = mass


def run_tests():
    print("Running BeamProgressScorer mock tests...")

    # Grid and mock BFM
    x = np.linspace(0.0, 100.0, 11)
    y = np.linspace(0.0, 50.0, 6)
    z = np.linspace(0.0, 50.0, 6)
    bfm = MockBFM(x, y, z)

    species = Species(mass=28.0)

    # Test 1: all particles travel straight to target (x=0) with sufficient speed
    N = 10
    start_point = [50.0, 25.0, 25.0]
    start_positions = np.tile(start_point, (N, 1))
    # velocities toward negative x, 100 mm/s
    start_velocities = np.zeros((N, 3))
    start_velocities[:, 0] = -200.0  # mm/s

    # Build mocks that BeamProgressScorer expects
    Beam = MockBeam
    Trajectory = MockTrajectory
    RK4Integrator = MockRK4Integrator

    scorer = BeamProgressScorer(
        bfm=bfm,
        species=species,
        start_positions=start_positions,
        start_velocities=start_velocities,
        dt=0.01,          # seconds
        num_steps=100,    # enough to reach
        Beam=Beam,
        Trajectory=Trajectory,
        RK4Integrator=RK4Integrator,
        travel_axis=0,
        travel_direction=-1,
    )

    # For the mock integrator to work we need to ensure the trajectory object gets velocities.
    # We'll monkeypatch the Trajectory class's __init__ to attach velocities from the Beam instance.
    orig_traj_init = MockTrajectory.__init__

    def patched_traj_init(self, beam_obj):
        orig_traj_init(self, beam_obj)
        # attach velocities so integrator can read them
        self.velocities = beam_obj.velocity

    MockTrajectory.__init__ = patched_traj_init

    # Now call scorer
    voltages = np.zeros(19)
    res = scorer.score(voltages)
    print("Test 1 - all reach: mean_progress=", res["mean_progress"], "survival_rate=", res["survival_rate"])

    # Restore trajectory init
    MockTrajectory.__init__ = orig_traj_init

    # Test 2: particles with lateral velocity that exit bbox quickly
    start_velocities2 = np.zeros((N, 3))
    start_velocities2[:, 0] = -10.0   # slow forward
    start_velocities2[:, 1] = 100.0   # large lateral -> leave bbox

    scorer2 = BeamProgressScorer(
        bfm=bfm,
        species=species,
        start_positions=start_positions,
        start_velocities=start_velocities2,
        dt=0.01,
        num_steps=100,
        Beam=Beam,
        Trajectory=Trajectory,
        RK4Integrator=RK4Integrator,
        travel_axis=0,
        travel_direction=-1,
    )

    # patch again
    MockTrajectory.__init__ = patched_traj_init
    res2 = scorer2.score(np.zeros(19))
    print("Test 2 - many lost: mean_progress=", res2["mean_progress"], "survival_rate=", res2["survival_rate"])
    MockTrajectory.__init__ = orig_traj_init

    # Test 3: partial progress (not enough steps to reach)
    start_velocities3 = np.zeros((N, 3))
    start_velocities3[:, 0] = -0.5  # mm/s, very slow

    scorer3 = BeamProgressScorer(
        bfm=bfm,
        species=species,
        start_positions=start_positions,
        start_velocities=start_velocities3,
        dt=0.1,
        num_steps=10,
        Beam=Beam,
        Trajectory=Trajectory,
        RK4Integrator=RK4Integrator,
        travel_axis=0,
        travel_direction=-1,
    )

    MockTrajectory.__init__ = patched_traj_init
    res3 = scorer3.score(np.zeros(19))
    print("Test 3 - partial progress: mean_progress=", res3["mean_progress"], "survival_rate=", res3["survival_rate"])
    MockTrajectory.__init__ = orig_traj_init

    print("\nAll mock tests completed.")


if __name__ == "__main__":
    run_tests()
