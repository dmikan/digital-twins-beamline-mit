"""
BeamProgressScorer
===================

A physics-based scoring filter for candidate electrode voltage configurations
(e.g. ones proposed by Optuna). For a FIXED beam of particles, it scores a
voltage configuration by how far the beam progresses through the setup before
either (a) reaching the target exit plane, or (b) being lost -- exiting the
chamber sideways or backward, i.e. hitting a wall/electrode.

HOW TO USE
----------
Works whether you paste this into the same file as your class definitions,
or import it as a separate module -- the Beam / Trajectory / RK4Integrator
classes are passed in explicitly rather than assumed to be in scope.

    from RK4_sim_basis import BasisFieldMap, IonSpecies, Beam, Trajectory, RK4Integrator
    from beam_progress_filter import make_beam, BeamProgressScorer

    bfm = BasisFieldMap.from_directory("./", n_electrodes=19)
    species = IonSpecies(mass=28*u, charge=1*e)

    # Build ONE fixed beam and reuse it for every voltage configuration you
    # score -- this is what makes scores comparable across Optuna trials.
    start_positions, start_velocities = make_beam(
        N=500, species=species, start_point=[395.0, 75.0, 77.0],
        mean_energy_eV=15.0, std_energy_eV=0.42466, half_angle_deg=15.0,
        seed=42,
    )

    scorer = BeamProgressScorer(
        bfm=bfm, species=species,
        start_positions=start_positions, start_velocities=start_velocities,
        dt=5e-8, num_steps=5000,
        Beam=Beam, Trajectory=Trajectory, RK4Integrator=RK4Integrator,
    )

    result = scorer.score(voltages)   # voltages: length-19 array
    print(result["mean_progress"])    # 0..1, primary scalar for Optuna

To score against an actual detector volume instead of a single exit plane,
pass detector_bbox -- "success" then means the particle's position falls
inside this 3D box (all of x, y, and z at once), not just crossing one
coordinate on travel_axis:

    scorer = BeamProgressScorer(
        bfm=bfm, species=species,
        start_positions=start_positions, start_velocities=start_velocities,
        dt=5e-8, num_steps=5000,
        Beam=Beam, Trajectory=Trajectory, RK4Integrator=RK4Integrator,
        detector_bbox=(0.0, 15.0, 60.0, 90.0, 60.0, 90.0),  # xmin,xmax,ymin,ymax,zmin,zmax
    )
"""

import numpy as np


# ----------------------------------------------------------------------
# Fixed-beam generator (factored out of your existing beam-setup code so
# it can be seeded and reused identically across every scored configuration)
# ----------------------------------------------------------------------
def make_beam(N, species, start_point, mean_energy_eV, std_energy_eV,
              half_angle_deg, cone_axis=(-1.0, 0.0, 0.0), seed=None):
    """
    Generate N particle initial conditions: a Gaussian energy spread and
    a cone of directions around `cone_axis`, matching the beam-setup logic
    already in your script. Pass a fixed `seed` so the SAME beam is
    reproduced every time -- required for fair comparison across
    voltage configurations.

    Returns
    -------
    start_positions, start_velocities : ndarray, shape (N, 3)
    """
    rng = np.random.default_rng(seed)

    e = 1.602176634e-19  # C

    start_positions = np.zeros((N, 3))
    start_positions[:] = start_point

    energies = rng.normal(mean_energy_eV, std_energy_eV, N)
    energies = np.clip(energies, 0, None)
    energies *= e  # -> Joules

    half_angle = np.deg2rad(half_angle_deg)
    cos_theta = rng.uniform(np.cos(half_angle), 1.0, N)
    theta = np.arccos(cos_theta)
    phi = rng.uniform(0, 2 * np.pi, N)

    axis = np.asarray(cone_axis, dtype=float)
    axis /= np.linalg.norm(axis)

    # Builds directions in a frame where `axis` is the cone's central
    # direction. Assumes axis is along a coordinate axis (matching the -x
    # setup). If you change cone_axis to something not aligned with a
    # coordinate axis, this needs a proper rotation instead.
    directions = np.empty((N, 3))
    directions[:, 0] = axis[0] * np.cos(theta)
    directions[:, 1] = np.sin(theta) * np.cos(phi)
    directions[:, 2] = np.sin(theta) * np.sin(phi)
    directions /= np.linalg.norm(directions, axis=1)[:, None]

    speeds = 1000 * np.sqrt(2 * energies / species.mass)  # mm/s, matches your script
    start_velocities = speeds[:, None] * directions

    return start_positions, start_velocities


# ----------------------------------------------------------------------
# The scorer
# ----------------------------------------------------------------------
class BeamProgressScorer:
    def __init__(self, bfm, Trajectory, dt, num_steps,
                 detector_bbox=None, travel_axis=0, travel_direction=-1,
                 target=None, bbox=None, wall_index=None, wall_hit_margin=1.0,
                 wall_check_midpoints=True, wall_check_stride=1,
                 terminate_on_wall_hit=False):
        """
        Parameters
        ----------
        bfm : BasisFieldMap
            The field map for the beam.
        Trajectory : Trajectory
            The trajectory of the beam particles.
        detector_bbox : tuple, optional
            (x_min, x_max, y_min, y_max, z_min, z_max) of the actual
            detector volume you want ions to reach.
        wall_index : object with .distance(points) -> ndarray, optional
            Fast approximate nearest-electrode-surface-distance query, e.g.
            an electrode_geometry.WallIndex built from the electrode STLs
            (already in workbench coordinates -- see that module). When
            given, score() additionally flags particles whose trajectory
            comes within `wall_hit_margin` of an electrode surface at any
            point before they reach the target or leave the outer bbox --
            something the outer-bbox check alone can't see, since
            electrodes sit well inside the chamber. Left as a duck-typed
            object (not importing trimesh/scipy here) so this module keeps
            its only hard dependency on numpy.
        wall_hit_margin : float
            Distance (mm) from an electrode surface counted as "hit it".
            Only used when wall_index is given. Should be set no smaller
            than the wall_index's own sampling resolution (its
            target_spacing), since distances tighter than that aren't
            meaningfully resolved.
        wall_check_midpoints : bool
            Also check the midpoint of every step's segment (catches a
            particle passing through a thin electrode between two recorded
            steps), on top of checking every recorded position. This is
            the dominant cost of scoring (profiled: ~86% of a
            combined_score() call at screening fidelity is inside the wall
            check, and this doubles how many points it queries) -- set
            False to skip it when you want the cheaper, coarser signal
            (e.g. RK4 screening, not final validation).
        wall_check_stride : int
            Only wall-check every Nth step (plus the very first and very
            last active step, so a particle's early/final position is
            never skipped) instead of every step. Cuts the wall-check
            point count by ~stride regardless of wall_check_midpoints.
            1 = check every step (default, most thorough).
        terminate_on_wall_hit : bool
            When True (and wall_index is given), a particle's trajectory
            is TERMINATED at its first wall contact for all scoring
            purposes: it can no longer count as reached/lost afterward,
            and its target-distance window ends at the hit step. This is
            what real SIMION does (the ion splats) -- without it, a
            particle that grazes an electrode keeps flying in RK4 and can
            still earn full detector-proximity credit, which measured out
            as RK4 predicting ~50% survival where SIMION delivers ~1%.
            Default False to preserve the original flag-only behavior for
            existing tools.
        dt : float
            RK4 time step (seconds).
        num_steps : int
            Steps to integrate. Particles still in-bounds and short of the
            target when this runs out get partial credit for how far they got.
            reuse across every voltage configuration you score.
        dt : float
            RK4 time step (seconds).
        num_steps : int
            Steps to integrate. Particles still in-bounds and short of the
            target when this runs out get partial credit for how far they got.
        Beam, Trajectory, RK4Integrator : classes
            Pass your existing class definitions in directly (dependency
            injection) so this module doesn't depend on global scope --
            works whether pasted inline or imported separately.
        travel_axis : int
            0/1/2 for x/y/z, the axis the beam nominally travels along.
            Defaults to 0 (x), matching your existing setup.
        travel_direction : int
            -1 or +1. -1 = progress means decreasing coordinate (matches your
            beam launching toward -x).
        target : float, optional
            Coordinate value counted as "successfully traversed the setup".
            Defaults to the bounding-box edge in the direction of travel.
        bbox : tuple, optional
            (x_min, x_max, y_min, y_max, z_min, z_max). Defaults to the
            extent of bfm's sampled grid. This is the OUTER chamber wall --
            exiting it sideways/backward counts as "lost".
        detector_bbox : tuple, optional
            (x_min, x_max, y_min, y_max, z_min, z_max) of the actual
            detector volume you want ions to reach. When given, "success"
            means the particle's position falls INSIDE this box on all
            three axes at once, not just crossing a single plane on
            travel_axis. Independent of `bbox` -- can be any region inside
            (or at the edge of) the chamber. If None (default), falls back
            to the original single-plane behavior.
        """

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
                # Scalar target used only for the partial-credit progress
                # fraction (see score()) -- the near face of the detector,
                # i.e. whichever face the beam reaches first. Actual success
                # is decided by full 3D box membership, not this plane.
                target = self._det_hi[travel_axis] if travel_direction < 0 else self._det_lo[travel_axis]
            else:
                target = lo[travel_axis] if travel_direction < 0 else hi[travel_axis]
        self.target = target

        self.start_coord = self.start_positions[:, travel_axis]

    def score(self, voltages):
        """
        Set the given voltage vector, fly the FIXED beam, and score how far
        each particle progressed toward `target`.

        Returns a dict with:
            mean_progress   : float in [0, 1] -- primary scalar score.
                              1.0 = every particle fully reached the target.
            survival_rate   : fraction of particles that reached target
                              without being lost first.
            progress        : ndarray (N,), per-particle progress in [0, 1]
            reached_target  : ndarray (N,) bool
            lost            : ndarray (N,) bool
            lost_step       : ndarray (N,) int (-1 if never lost)
            voltages        : the voltage vector that was scored
        """


        # (T, N, 3) -- T = num_steps + 1, includes the initial state
        states = np.asarray(self._Trajectory.states)  # (T, N, 3)
        # print("Trajectory states shape:", states.shape)
        positions = np.array([s.position for s in states]) # (T, N, 3)
        print("Positions shape:", positions.shape)
        coord = positions[:, :, self.travel_axis]  # (T, N)

        # Reached target: either entered the 3D detector volume (if
        # detector_bbox was given), or crossed the target plane along
        # travel_axis (original single-plane behavior).
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

        # Lost: left the bounding box any OTHER way -- sideways (hit a wall)
        # or backward out of the entrance.
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

        # Step at which each particle is first resolved (reached OR lost),
        # else T-1 if neither happened within num_steps. Shared by the
        # wall-hit check below and by combined_score()'s target-distance
        # calculation -- both only care about the trajectory while it's
        # still physically "in flight".
        stopped = reached_mask | lost_mask
        never_stopped = ~stopped.any(axis=0)
        stop_idx = np.where(never_stopped, T - 1, np.argmax(stopped, axis=0))  # (N,)

        # Electrode-wall hits: the outer bbox check above only catches
        # particles leaving the chamber -- it's blind to particles that hit
        # an electrode while still well inside the chamber. Checked
        # separately here (only if a wall_index was supplied) and reported
        # alongside reached/lost rather than folded into them, since which
        # should "win" when they disagree is a scoring-policy decision, not
        # a geometry one.
        hit_wall = np.zeros(N, dtype=bool)
        hit_wall_step = np.full(N, -1, dtype=int)
        if self.wall_index is not None:
            hit_wall_mask = self._wall_hit_mask(positions, stop_idx)
            any_hit = hit_wall_mask.any(axis=0)
            hit_wall = any_hit
            hit_wall_step = np.where(any_hit, np.argmax(hit_wall_mask, axis=0), -1)

            if self.terminate_on_wall_hit and any_hit.any():
                # The particle splats at its first wall contact (matching
                # real SIMION): erase any reached/lost events from that
                # step onward and re-resolve stop_idx so the rest of the
                # scoring (progress loop below, combined_score()'s
                # target-distance window) never sees the post-hit flight.
                steps_col = np.arange(T)[:, None]                        # (T, 1)
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

        span = self.target - self.start_coord  # (N,)

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
                # Neither reached nor lost -- ran out of integration steps.
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
        # Exposed so other methods (e.g. combined_score()) can reuse the
        # already-integrated trajectory and the "still in flight" window
        # without recomputing them.
        result["positions"] = positions
        result["stop_idx"] = stop_idx
        return result

    def _wall_hit_mask(self, positions, stop_idx):
        """
        (T, N) bool: whether the particle is within wall_hit_margin of an
        electrode surface at step t, checked at every KEPT recorded
        position (see wall_check_stride) AND, if wall_check_midpoints, at
        the midpoint of the segment arriving at every kept step (to catch
        a particle passing close to/through a thin electrode between two
        recorded steps, not just at the sampled endpoints).

        Only checked up to each particle's earliest reached/lost step
        (inclusive, given by stop_idx) -- once a particle is already
        resolved, its continued-but-physically-moot trajectory isn't worth
        querying, and skipping it is the main lever for keeping this
        affordable over a full 500-particle x 5000-step trajectory (see
        electrode_geometry.py for the wall_index speed tradeoffs).

        Profiled: this is the dominant cost of scoring (~86% of a
        combined_score() call at screening fidelity), almost entirely
        proportional to how many points get queried -- wall_check_stride
        and wall_check_midpoints=False are the two knobs that actually
        move that number, tried and measured faster than micro-optimizing
        the query itself (vectorizing the AABB cull loop in
        electrode_geometry.WallIndex, or using float32, both measured
        under ~15% improvement -- not worth the added complexity/risk).
        """
        T, N = positions.shape[0], positions.shape[1]

        step_idx = np.arange(T)[:, None]
        active_mask = step_idx <= stop_idx[None, :]  # (T, N), True while still "in flight"

        if self.wall_check_stride > 1:
            # Always keep step 0 (0 % stride == 0) and each particle's own
            # final active step (stop_idx), so striding can't skip a
            # particle's very first or very last position.
            stride_mask = (step_idx % self.wall_check_stride == 0)
            final_step_mask = (step_idx == stop_idx[None, :])
            keep_mask = active_mask & (stride_mask | final_step_mask)
        else:
            keep_mask = active_mask

        hit_wall_mask = np.zeros((T, N), dtype=bool)

        # Endpoints
        tt, pp = np.nonzero(keep_mask)
        pts = positions[tt, pp]
        if len(pts) > 0:
            dist = self.wall_index.distance(pts)
            hit = dist <= self.wall_hit_margin
            hit_wall_mask[tt[hit], pp[hit]] = True

        # Segment midpoints -- gated by the ARRIVAL step (t+1) being kept,
        # not by striding the segments themselves (a kept step's incoming
        # segment is still checked in full, only which steps count as
        # "kept" is reduced).
        if T > 1 and self.wall_check_midpoints:
            mids = 0.5 * (positions[:-1] + positions[1:])  # (T-1, N, 3)
            arrival_kept = keep_mask[1:]  # (T-1, N)
            mt, mp = np.nonzero(arrival_kept)
            mpts = mids[mt, mp]
            if len(mpts) > 0:
                mdist = self.wall_index.distance(mpts)
                mhit = mdist <= self.wall_hit_margin
                hit_wall_mask[mt[mhit] + 1, mp[mhit]] = True

        return hit_wall_mask

    def _target_distance(self, positions, stop_idx):
        """
        (N,) mm: closest approach of each particle's trajectory to
        detector_bbox (0 if it ever entered the box), over the same
        "still in flight" window used by _wall_hit_mask (0..stop_idx
        inclusive) -- distance after a particle is already resolved isn't
        meaningful to the score.

        Plain axis-aligned point-to-box distance (0 inside, else Euclidean
        distance to the nearest face/edge/corner) -- cheap numpy, no
        wall_index/mesh involved, this is against the target region, not
        the electrodes.
        """
        T, N = positions.shape[0], positions.shape[1]
        det_lo = np.asarray(self._det_lo)
        det_hi = np.asarray(self._det_hi)

        # In-place accumulation: below+above as a separate temporary was a
        # third (T,N,3) float64 array live at once -- at rescreen fidelity
        # that's an extra ~200MB and was the allocation that tipped a
        # memory-tight run over (MemoryError, 2026-07-05).
        below = np.maximum(det_lo - positions, 0.0)  # (T, N, 3)
        below += np.maximum(positions - det_hi, 0.0)
        dist = np.linalg.norm(below, axis=2)  # (T, N)

        step_idx = np.arange(T)[:, None]
        active_mask = step_idx <= stop_idx[None, :]
        dist = np.where(active_mask, dist, np.inf)
        return dist.min(axis=0)  # (N,)

    def combined_score(self, voltages, target_weight=1.0, wall_weight=1.0,
                        lost_weight=0.3, target_scale=150.0):
        """
        A single scalar for comparing/ranking REGIONS of voltage space --
        not a precise objective to converge on (that's what a real,
        SIMION-backed optimizer run is for). Meant to be cheap enough to
        run over many Optuna-suggested voltage arrays at once (see
        voltage_batch_filter.py) so obviously-bad regions can be ruled out,
        and survivors' scores can feed the GP as extra, SIMION-free signal
        -- particularly useful in regions where the GP itself is uncertain.

        Combines three signals as a weighted sum:

          + target_weight * mean(target_reward)
                target_reward = exp(-target_distance / target_scale), in
                (0, 1]. target_distance is each particle's closest 3D
                approach to the ACTUAL detector volume (detector_bbox),
                not just whether it crossed a plane on travel_axis -- this
                is the fix for score()'s progress being blind to y/z miss
                distance (see the class-level scoring explanation). A beam
                that misses the detector by 2mm now scores meaningfully
                higher than one that misses by 200mm.

          - wall_weight * wall_hit_fraction
                Fraction of particles that came within wall_hit_margin of
                an electrode surface (see wall_index / _wall_hit_mask).

          - lost_weight * lost_fraction
                Fraction that left the outer chamber cleanly (no electrode
                involved). Weighted less than a wall hit by default --
                exiting the chamber is a softer failure (mis-aim) than
                hitting solid geometry, but you may want to tune this.

        Requires both detector_bbox and wall_index to be set on this
        scorer -- this method only makes sense as the sum of those two
        checks plus the target reward.

        Parameters
        ----------
        target_scale : float, mm
            Distance over which target_reward decays by 1/e. WAS 30mm
            (picked to match the lens/bender feature size) but that's the
            wrong length scale to calibrate against -- it's the DISTANCE
            TO THE DETECTOR that needs to vary the reward, and real
            candidates measured this session land 200-350mm away (mean
            323mm, best-of-batch 235mm). At scale=30, exp(-235/30) through
            exp(-350/30) are all < 0.0005 -- indistinguishable at the
            precision combined_score is printed/compared at, which is
            exactly why every batch this session showed identical
            combined_score to 3 decimals regardless of voltage draw: this
            term contributed zero usable signal across the entire searched
            region, leaving wall_hit_fraction/lost_fraction (which also
            cluster tightly) to dominate a basically-flat sum. 150mm keeps
            exp(-d/scale) in a discriminating 0.1-0.45 range across the
            200-350mm region actually being sampled, while still decaying
            toward 0 for genuinely hopeless candidates (500mm+).

        Returns
        -------
        Everything score() returns, plus:
            target_distance : (N,) mm
            target_reward   : (N,) in (0, 1]
            combined_score  : float
        """
        if self.detector_bbox is None:
            raise ValueError(
                "combined_score needs detector_bbox set to the real detector "
                "volume (e.g. optimizer.py's DETECTOR_REGION, in workbench mm: "
                "x=(70,82), y=(70,83), z=(403,407)) -- a single-plane target "
                "has no y/z information to compute a 3D distance reward from."
            )
        if self.wall_index is None:
            raise ValueError(
                "combined_score needs wall_index set (see electrode_geometry."
                "build_wall_index) -- without it there's no collision penalty "
                "to combine the target reward with."
            )

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