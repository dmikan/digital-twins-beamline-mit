"""
RK4_sim_basis_batch
=====================

Batched extension of RK4_sim_basis.py -- does NOT modify that file (kept as
backup / current single-config code). Solves this problem:

    BasisFieldMap.set_voltages(voltages) OVERWRITES self.V / self.E / the
    interpolator in place -- it can only hold ONE voltage configuration's
    combined field at a time. Scoring M candidate voltage arrays (e.g.
    digital-twins-beamline-mit/sim.py's loop over Optuna-suggested arrays)
    therefore means M sequential calls to set_voltages() + a full
    RK4Integrator.integrate(), one config fully finished before the next
    one starts.

WHAT'S DIFFERENT HERE
----------------------
BatchBasisFieldMap.set_voltages_batch(voltages_batch) combines a WHOLE
BATCH of M voltage arrays at once (still just one linear-algebra call --
V_batch = tensordot(voltages_batch, basis_V)) and keeps all M combined
fields in memory simultaneously as one 4-D array (M, Nx, Ny, Nz, 3). Field
lookups are then done through a single 4-D interpolator built over
(config_index, x, y, z) -- config_index is an exact-integer grid axis, so
querying at an integer index returns that config's field exactly (no
cross-config blending).

That lets ALL M configs' particles be integrated in ONE RK4Integrator
call: tag each particle with which config it belongs to (make_batch_beam),
and BatchRK4Integrator looks up each particle's field from its own config
via field_batch(). Same total floating-point work as the sequential
version, but one Python-level integration loop (num_steps iterations)
instead of M of them, and one interpolator instead of M.

MEMORY TRADEOFF -- read this before picking M
------------------------------------------------
Holding all M fields at once costs M * Nx * Ny * Nz * 3 * 8 bytes. At this
project's grid (50x50x50) that's ~300MB for M=100, growing linearly with
M. If you need a much larger M, either lower the grid resolution (see
extract_PA.lua sample_count_x/y/z), pass dtype=np.float32 to
set_voltages_batch (halves memory), or process the candidates in
sub-batches (e.g. 100 at a time) rather than all at once.

USAGE
-----
    from RK4_sim_basis_batch import (
        BatchBasisFieldMap, make_batch_beam, BatchTrajectory,
        BatchRK4Integrator, split_batch_trajectory,
    )
    from RK4_sim_basis import IonSpecies  # unchanged, reused as-is

    bfm = BatchBasisFieldMap.from_directory("./", n_electrodes=19)
    bfm.set_voltages_batch(voltages_batch)   # (M, 19)

    beam, config_indices = make_batch_beam(species, start_positions, start_velocities, M)
    trajectory = BatchTrajectory(beam, config_indices)
    BatchRK4Integrator(bfm, config_indices).integrate(trajectory, dt, num_steps)

    # Hand each config's slice to the EXISTING, unmodified BeamProgressScorer:
    for sub_traj in split_batch_trajectory(trajectory, config_indices, M):
        scorer = BeamProgressScorer(bfm=bfm, Trajectory=sub_traj, dt=dt, num_steps=num_steps, ...)
        result = scorer.combined_score(voltages)
"""

from types import SimpleNamespace

import numpy as np
from scipy.ndimage import map_coordinates

from RK4_sim_basis import BasisFieldMap, ParticleState, Beam


class BatchBasisFieldMap(BasisFieldMap):
    """
    Same basis-file loading as BasisFieldMap (inherited, unmodified) plus
    set_voltages_batch()/field_batch() for handling M voltage
    configurations' combined fields at once. The inherited single-config
    set_voltages() still works normally (e.g. from_directory's __init__
    calls set_voltages(zeros) once).

    field() is OVERRIDDEN here (not just extended) to use
    scipy.ndimage.map_coordinates instead of BasisFieldMap's
    RegularGridInterpolator. Profiling the integrator found ~95% of total
    RK4 integration time was inside RegularGridInterpolator._evaluate_linear
    -- map_coordinates is a specialized, compiled implementation for exactly
    this case (interpolating on a REGULAR grid; RegularGridInterpolator's
    generality includes supporting non-uniform grids, which costs a binary
    search per axis per point that a uniform grid doesn't need). Benchmarked
    at ~2.2-2.65x faster, verified to match RegularGridInterpolator's output
    to ~7e-6 (float rounding, not a real behavior difference). Because this
    override lives on BatchBasisFieldMap (not BasisFieldMap itself), the
    ORIGINAL RK4_sim_basis.py / RK4Integrator are untouched -- you get the
    speedup by using BatchBasisFieldMap in place of BasisFieldMap, batched
    or not (RK4Integrator just calls field_map.field(...) polymorphically).
    """

    def field(self, positions):
        """Same contract as BasisFieldMap.field -- see class docstring for why
        this is a faster reimplementation rather than an extension."""
        positions = np.asarray(positions)
        if positions.shape[-1] != 3:
            raise ValueError("positions must have shape (..., 3)")
        original_shape = positions.shape[:-1]
        positions = positions.reshape(-1, 3)

        clipped = np.array(positions, copy=True)
        clipped[:, 0] = np.clip(clipped[:, 0], self.x.min(), self.x.max())
        clipped[:, 1] = np.clip(clipped[:, 1], self.y.min(), self.y.max())
        clipped[:, 2] = np.clip(clipped[:, 2], self.z.min(), self.z.max())

        frac = np.empty((3, clipped.shape[0]))
        frac[0] = (clipped[:, 0] - self.x[0]) / self.dx
        frac[1] = (clipped[:, 1] - self.y[0]) / self.dy
        frac[2] = (clipped[:, 2] - self.z[0]) / self.dz

        E = np.empty((clipped.shape[0], 3))
        for c in range(3):
            map_coordinates(self.E[..., c], frac, order=1, mode="nearest",
                             prefilter=False, output=E[:, c])

        outside = (
            (positions[:, 0] < self.x.min()) | (positions[:, 0] > self.x.max()) |
            (positions[:, 1] < self.y.min()) | (positions[:, 1] > self.y.max()) |
            (positions[:, 2] < self.z.min()) | (positions[:, 2] > self.z.max())
        )
        if np.any(outside):
            E[outside] = 0.0

        return E.reshape(*original_shape, 3)

    def set_voltages_batch(self, voltages_batch, dtype=np.float64):
        """
        Parameters
        ----------
        voltages_batch : (M, n_electrodes) array_like
        dtype : numpy dtype
            float32 halves memory vs the default float64, at the cost of
            some precision -- see the module docstring's memory tradeoff.

        Stores self.V_batch (M,Nx,Ny,Nz) and self.E_batch (M,Nx,Ny,Nz,3) --
        field_batch() interpolates directly from E_batch via
        map_coordinates (see class docstring), no separate interpolator
        object needed.
        """
        voltages_batch = np.asarray(voltages_batch, dtype=dtype)
        if voltages_batch.ndim != 2 or voltages_batch.shape[1] != self.n_electrodes:
            raise ValueError(
                f"voltages_batch must have shape (M, {self.n_electrodes}), "
                f"got {voltages_batch.shape}"
            )
        M = voltages_batch.shape[0]
        self.M = M
        self.voltages_batch = voltages_batch

        # (M, Nx, Ny, Nz) -- one linear-algebra call combines every config's
        # potential at once, same underlying math as the single-config
        # set_voltages (V = sum_i voltages[i] * basis_V[i]).
        basis_V = self.basis_V.astype(dtype, copy=False)
        self.V_batch = np.tensordot(voltages_batch, basis_V, axes=([1], [0]))

        # Gradient only along the spatial axes (1,2,3); axis 0 (config) is
        # left as a batch dimension -- np.gradient supports this directly.
        dVdx, dVdy, dVdz = np.gradient(self.V_batch, self.dx, self.dy, self.dz, axis=(1, 2, 3))
        self.E_batch = -np.stack((dVdx, dVdy, dVdz), axis=-1).astype(dtype, copy=False)  # (M,Nx,Ny,Nz,3)
        return self.E_batch

    def field_batch(self, config_indices, positions):
        """
        Electric field at each (config, position) pair, for the batch built
        by set_voltages_batch.

        Parameters
        ----------
        config_indices : (K,) int array_like, values in [0, M)
        positions : (K, 3) array_like

        Returns
        -------
        (K, 3) ndarray
        """
        config_indices = np.asarray(config_indices)
        positions = np.asarray(positions)
        if positions.shape[0] != config_indices.shape[0]:
            raise ValueError("config_indices and positions must have the same length")

        clipped = np.array(positions, copy=True)
        clipped[:, 0] = np.clip(clipped[:, 0], self.x.min(), self.x.max())
        clipped[:, 1] = np.clip(clipped[:, 1], self.y.min(), self.y.max())
        clipped[:, 2] = np.clip(clipped[:, 2], self.z.min(), self.z.max())

        # config_indices are exact integers (spacing 1 along axis 0 of
        # E_batch), so they're valid "fractional" indices as-is -- no
        # scaling needed, unlike the spatial axes.
        frac = np.empty((4, clipped.shape[0]))
        frac[0] = config_indices
        frac[1] = (clipped[:, 0] - self.x[0]) / self.dx
        frac[2] = (clipped[:, 1] - self.y[0]) / self.dy
        frac[3] = (clipped[:, 2] - self.z[0]) / self.dz

        E = np.empty((clipped.shape[0], 3))
        for c in range(3):
            map_coordinates(self.E_batch[..., c], frac, order=1, mode="nearest",
                             prefilter=False, output=E[:, c])

        outside = (
            (positions[:, 0] < self.x.min()) | (positions[:, 0] > self.x.max()) |
            (positions[:, 1] < self.y.min()) | (positions[:, 1] > self.y.max()) |
            (positions[:, 2] < self.z.min()) | (positions[:, 2] > self.z.max())
        )
        if np.any(outside):
            E[outside] = 0.0

        return E


def make_batch_beam(species, start_positions, start_velocities, n_configs):
    """
    Replicate a single-config N-particle beam across n_configs configs,
    tagging each replica with which config it belongs to. Total particles
    = n_configs * N -- this is the SAME fixed beam reused for every config
    (matches make_beam's "reuse across every voltage configuration you
    score" design, just replicated instead of regenerated so every config
    is compared on identical starting conditions).

    Returns
    -------
    beam : RK4_sim_basis.Beam, n_particles = n_configs * N
    config_indices : (n_configs * N,) int ndarray, which config each particle belongs to
    """
    n = start_positions.shape[0]
    all_positions = np.tile(start_positions, (n_configs, 1))
    all_velocities = np.tile(start_velocities, (n_configs, 1))
    config_indices = np.repeat(np.arange(n_configs), n)
    beam = Beam(species=species, position=all_positions, velocity=all_velocities)
    return beam, config_indices


class BatchTrajectory:
    """Same shape/contract as RK4_sim_basis.Trajectory, plus config_indices."""

    def __init__(self, beam, config_indices):
        self.species = beam.species
        self.initial_state = beam.state
        self.states = [self.initial_state]
        self.config_indices = np.asarray(config_indices, dtype=int)

    def add_state(self, state):
        self.states.append(state)


# The whole pipeline works in millimeter units: the field grid is in mm
# with potentials in V, so interpolated E comes out in V/mm; positions and
# velocities are mm and mm/s (see beam_progress_score.make_beam's
# speeds = 1000*sqrt(...)). a = qE/m is only valid in SI: converting
# E V/mm -> V/m costs x1000, converting the resulting m/s^2 -> mm/s^2
# costs another x1000, so the acceleration in mm-units needs x1e6 overall.
# Without it every electric force is a million times too weak -- verified
# to be why voltage arrays differing by +-1000V produced RK4 trajectories
# identical to one decimal all session (SESSION_HANDOFF.txt finding 5(b)).
# NOTE: RK4_sim_basis.RK4Integrator (the kept-as-backup single-config
# path) still has the un-fixed formula.
E_VMM_TO_ACCEL_MM = 1.0e6


class BatchRK4Integrator:
    """
    Same RK4 stepping as RK4_sim_basis.RK4Integrator, but looks up each
    particle's field from ITS OWN config via field_map.field_batch()
    instead of a single shared field() -- this is what lets one
    integrate() call fly every config's particles together.
    """

    def __init__(self, field_map, config_indices):
        self.field_map = field_map
        self.config_indices = np.asarray(config_indices, dtype=int)

    def integrate(self, trajectory, dt, num_steps):
        state = trajectory.initial_state
        species = trajectory.species
        for _ in range(num_steps):
            k1 = self._derivative(state, species)
            k2 = self._derivative(ParticleState(state.position + 0.5 * dt * k1[0],
                                                 state.velocity + 0.5 * dt * k1[1]), species)
            k3 = self._derivative(ParticleState(state.position + 0.5 * dt * k2[0],
                                                 state.velocity + 0.5 * dt * k2[1]), species)
            k4 = self._derivative(ParticleState(state.position + dt * k3[0],
                                                 state.velocity + dt * k3[1]), species)

            new_position = state.position + (dt / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            new_velocity = state.velocity + (dt / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])

            state = ParticleState(new_position, new_velocity)
            trajectory.add_state(state)

    def _derivative(self, state, species):
        E_field = self.field_map.field_batch(self.config_indices, state.position)
        acceleration = species.charge * E_field / species.mass * E_VMM_TO_ACCEL_MM
        return (state.velocity, acceleration)


def split_batch_trajectory(batch_trajectory, config_indices, n_configs):
    """
    Split a BatchTrajectory's states back into n_configs separate
    Trajectory-LIKE objects (duck-typed: .species, .initial_state, .states
    with .position per state), one per voltage config.

    NOT needed just to SCORE a batch -- BeamProgressScorer's internals are
    already vectorized across whatever particle axis they're given, and
    config-independent state (bfm/wall_index/detector_bbox) is shared, so
    scoring the whole batch in ONE call and reshaping the resulting
    per-particle arrays to (n_configs, N) is both faster and avoids this
    function's memory cost (it makes n_configs FULL duplicate copies of the
    trajectory data, all held at once) -- see sim_batch.py's score_chunk for
    that approach, which is what the batch pipeline actually uses.

    This is still useful on its own for pulling out ONE specific config's
    Trajectory for inspection/plotting (e.g. reusing test_collision_
    filter.py's plotting code on a single candidate from a batch run).

    Returns
    -------
    list of n_configs SimpleNamespace objects
    """
    config_indices = np.asarray(config_indices)
    results = []
    for m in range(n_configs):
        mask = config_indices == m
        sub_states = [ParticleState(s.position[mask], s.velocity[mask]) for s in batch_trajectory.states]
        results.append(SimpleNamespace(
            species=batch_trajectory.species,
            initial_state=sub_states[0],
            states=sub_states,
        ))
    return results
