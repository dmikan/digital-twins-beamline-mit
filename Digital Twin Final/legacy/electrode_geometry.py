"""
electrode_geometry
===================

Loads the electrode_N.stl solid geometry and puts it into the SAME
coordinate frame as the basis field grid / RK4 trajectories (SIMION
workbench coordinates), so trajectory positions can be checked against
the actual electrode surfaces instead of only the outer bounding box.

WHY THE TRANSFORM
------------------
extract_PA.lua samples the field directly in workbench coordinates
(x: 0-484, y: 0-153, z: 0-484 mm -- see its x_min/x_max etc). The STL
files, however, were exported in a different frame -- not just offset, but
rotated a few degrees relative to workbench space (confirmed once actual
SIMION-workbench coordinates were available for several landmarks; a pure
translation, which is all an axis-aligned-bounding-box comparison can
detect, looked deceptively close but was off by 10-20mm per point).

The current transform is a proper rigid-body fit (rotation + translation,
via the Kabsch/SVD algorithm -- see fit_stl_to_workbench_transform) to 6
known correspondences: electrode 1 (source) and electrode 19 (Detector)
against RK4_sim_basis.py's/optimizer.py's known points, plus the four
quadrupole-bender electrodes (9, 10, 11, 12), whose workbench coordinates
were read directly off the SIMION GUI's cursor position readout. Residuals
after the fit are a consistent ~11-15mm across all 6 points (no outliers),
consistent with manual cursor-reading precision rather than a remaining
systematic error.

    workbench_xyz = stl_xyz @ STL_TO_WORKBENCH_ROTATION.T + STL_TO_WORKBENCH_TRANSLATION

If the geometry is ever re-exported, or more/better landmark points become
available, redo the fit with fit_stl_to_workbench_transform() -- don't
hand-tune these constants.

USAGE
-----
    from electrode_geometry import load_electrode_meshes, load_walls_mesh, build_wall_index

    meshes = load_electrode_meshes("./")        # dict {electrode_number: trimesh.Trimesh}
    walls = load_walls_mesh("./", exclude=(1, 19))  # merged mesh, skip source + Detector

    # FAST approximate distance from trajectory points to the nearest
    # electrode surface -- see WallIndex below for why this exists instead
    # of exact mesh queries:
    wall_index = build_wall_index("./", exclude=(1, 19))
    dist = wall_index.distance(points)  # points: (N, 3) -> (N,) mm
"""

from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

# Rigid-body fit (rotation + translation) from STL export space into SIMION
# workbench space -- see module docstring for how these were derived and
# why a translation alone (the original approach) wasn't good enough.
STL_TO_WORKBENCH_ROTATION = np.array([
    [ 0.99738233, -0.06972016, -0.01770672],
    [ 0.07079483,  0.99522571,  0.06696917],
    [ 0.01297262, -0.06805272,  0.99760373],
])
STL_TO_WORKBENCH_TRANSLATION = np.array([311.72967676, 164.97727664, 175.84421467])


def _stl_to_workbench_matrix(rotation=STL_TO_WORKBENCH_ROTATION, translation=STL_TO_WORKBENCH_TRANSLATION):
    """4x4 homogeneous transform combining rotation + translation, for trimesh.apply_transform."""
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def fit_stl_to_workbench_transform(stl_points, workbench_points):
    """
    Least-squares rigid-body fit (Kabsch/SVD algorithm) mapping a set of
    known STL-space points to their corresponding known workbench-space
    points. Use this to (re)derive STL_TO_WORKBENCH_ROTATION/TRANSLATION
    from landmark correspondences (e.g. electrode centroids matched against
    coordinates read off the SIMION GUI) -- at least 3 non-collinear points
    are required; more gives a more robust fit and lets you sanity-check
    per-point residuals for outliers/typos.

    Parameters
    ----------
    stl_points, workbench_points : (N, 3) array_like, N >= 3, corresponding rows

    Returns
    -------
    rotation : (3, 3) ndarray
    translation : (3,) ndarray
    residuals : (N,) ndarray, mm -- per-point fit error, for sanity-checking
    """
    src = np.asarray(stl_points, dtype=float)
    dst = np.asarray(workbench_points, dtype=float)
    src_c, dst_c = src.mean(axis=0), dst.mean(axis=0)
    H = (src - src_c).T @ (dst - dst_c)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = dst_c - R @ src_c
    residuals = np.linalg.norm((R @ src.T).T + t - dst, axis=1)
    return R, t, residuals


def find_stl_to_workbench_offset(directory, pattern="electrode_{i}.stl", n_electrodes=19, start_index=1):
    """
    SUPERSEDED by the rigid-body fit above (this only recovers a
    translation, not the rotation -- kept for reference/comparison only).
    Translates the combined bounding-box minimum corner of all electrode
    STLs to (0, 0, 0).
    """
    directory = Path(directory)
    mins = []
    for i in range(start_index, start_index + n_electrodes):
        m = trimesh.load(directory / pattern.format(i=i))
        mins.append(m.bounds[0])
    mins = np.array(mins)
    return -mins.min(axis=0)


def load_electrode_meshes(directory, pattern="electrode_{i}.stl", n_electrodes=19, start_index=1,
                           rotation=STL_TO_WORKBENCH_ROTATION, translation=STL_TO_WORKBENCH_TRANSLATION):
    """
    Load each electrode_N.stl and transform it into workbench coordinates
    (rotation + translation -- see module docstring).

    Returns
    -------
    dict {electrode_number (1-based int): trimesh.Trimesh}
    """
    directory = Path(directory)
    transform = _stl_to_workbench_matrix(rotation, translation)
    meshes = {}
    for i in range(start_index, start_index + n_electrodes):
        fpath = directory / pattern.format(i=i)
        if not fpath.exists():
            raise FileNotFoundError(f"Expected electrode STL not found: {fpath}")
        mesh = trimesh.load(fpath)
        mesh.apply_transform(transform)
        meshes[i] = mesh
    return meshes


def load_walls_mesh(directory, exclude=(), **kwargs):
    """
    Load all electrode STLs (in workbench coordinates) and merge them into
    a single mesh representing the physical "walls" a particle can hit.

    Parameters
    ----------
    exclude : iterable of int
        Electrode numbers to leave out of the merged mesh -- e.g. exclude
        the Detector (19) if landing there should NOT count as hitting a
        wall.
    **kwargs : forwarded to load_electrode_meshes (pattern, n_electrodes, ...)

    Returns
    -------
    trimesh.Trimesh
    """
    meshes = load_electrode_meshes(directory, **kwargs)
    kept = [m for num, m in meshes.items() if num not in exclude]
    if not kept:
        raise ValueError("exclude removed every electrode -- nothing left to merge")
    return trimesh.util.concatenate(kept)


# ----------------------------------------------------------------------
# Fast approximate wall-distance queries
# ----------------------------------------------------------------------
#
# Scoring a trajectory means checking distance-to-wall at up to
# n_particles * num_steps points (500 x 5000 = 2.5M for the default beam).
# Exact mesh queries are far too slow at that scale -- benchmarked against
# this project's merged electrode mesh (~3500 triangles):
#
#   trimesh ray casting (mesh.ray.intersects_location)  ~300 us/query
#   trimesh exact proximity (mesh.nearest.on_surface)   ~450 us/query
#
# Both extrapolate to 750-1200 SECONDS for one full scorer.score() call --
# unusable inside an Optuna loop. Naive uniform subdivision (subdivide
# every triangle until short edges are all < max_edge) is also a trap here:
# a few huge flat backing-plate triangles (up to 484 mm edges) blow up to
# tens of millions of triangles before the small detail triangles are even
# affected -- it ran out of memory in testing.
#
# WallIndex instead area-weight-samples the mesh SURFACE ONCE into a dense
# point cloud (trimesh.sample.sample_surface, which handles mixed triangle
# sizes fine) and answers distance queries with a scipy cKDTree nearest-
# neighbor lookup. That benchmarks at ~1-30 us/query depending on point
# cloud size -- a 10-300x speedup -- at the cost of a small, bounded
# approximation error (roughly the sample spacing; ~0.08 mm mean / ~2 mm
# worst case at the default 2 mm spacing, tested against exact queries).
# Good enough for a "is this beam grazing/hitting an electrode" scoring
# signal; NOT a substitute for exact collision geometry if you ever need
# that elsewhere.
class WallIndex:
    """Fast approximate nearest-wall-surface-distance queries via cKDTree."""

    def __init__(self, mesh, target_spacing=2.0, seed=0, electrode_boxes=None, aabb_pad=15.0):
        """
        Parameters
        ----------
        mesh : trimesh.Trimesh
            Wall geometry in workbench coordinates (e.g. from load_walls_mesh).
        target_spacing : float
            Approximate mm spacing between surface sample points. Smaller
            = more accurate distances, more points, slower queries/build.
            2.0 mm is a reasonable default relative to a ~0.5 mm per-step
            particle displacement in the example beam config.
        electrode_boxes : list of (2,3) arrays, optional
            Per-ELECTRODE (not merged-mesh) bounding boxes, e.g. from
            load_electrode_meshes -- used as a cheap broad-phase filter:
            points that aren't within aabb_pad of ANY individual electrode
            are guaranteed farther than aabb_pad from the actual surface
            (each electrode's mesh is entirely contained in its own box, so
            distance-to-box is a valid lower bound on distance-to-mesh) and
            are answered directly without touching the KDTree. This is the
            single biggest lever found while benchmarking: on this
            project's beamline the electrodes are compact and scattered
            through the chamber, so >95% of random chamber points get
            rejected this way -- EXCEPT the merged mesh's own bounding box
            can be misleading for broad-phase (e.g. this project's electrode
            2 is a full-length pipe/housing whose box spans the entire
            chamber), which is why boxes must be passed per-electrode, not
            computed from the merged mesh. Pass None to disable (falls back
            to always querying the KDTree directly).
        aabb_pad : float
            Padding (mm) applied to each electrode's box before culling.
            MUST be larger than any wall_hit_margin you intend to use --
            culled points are reported at exactly this distance (a safe
            lower bound, not the true distance), so if aabb_pad were
            smaller than your margin, a genuinely-far point could be
            reported as "closer than margin" and misclassified as a hit.
            Default 15mm has generous headroom over a typical ~1-2mm margin.
        """
        self.mesh = mesh
        self.target_spacing = target_spacing
        n_samples = max(1, int(mesh.area / target_spacing**2))
        surface_pts, _ = trimesh.sample.sample_surface(mesh, n_samples, seed=seed)
        # Surface sampling alone can under-represent sharp edges/corners
        # (it's area-weighted, not feature-weighted) -- add the original
        # mesh vertices too, which are cheap and cover exactly those spots.
        points = np.vstack([surface_pts, mesh.vertices])
        self.points = points
        self.tree = cKDTree(points)

        self.aabb_pad = aabb_pad
        if electrode_boxes:
            boxes = np.asarray(electrode_boxes)  # (n_electrodes, 2, 3)
            self._cull_lo = boxes[:, 0, :] - aabb_pad
            self._cull_hi = boxes[:, 1, :] + aabb_pad
        else:
            self._cull_lo = self._cull_hi = None

    def distance(self, points):
        """
        Nearest-wall-surface distance (mm) for each input point. Points
        culled by the AABB pre-filter (see electrode_boxes above) are
        reported as exactly aabb_pad -- a safe lower bound, not the true
        distance; fine for thresholding against a margin < aabb_pad, not
        for anything that needs the actual distance value.

        Parameters
        ----------
        points : (..., 3) array_like

        Returns
        -------
        ndarray, shape (...,)
        """
        points = np.asarray(points, dtype=float)
        original_shape = points.shape[:-1]
        flat = points.reshape(-1, 3)

        if self._cull_lo is None:
            dist, _ = self.tree.query(flat)
            return dist.reshape(original_shape)

        candidate = np.zeros(len(flat), dtype=bool)
        for lo, hi in zip(self._cull_lo, self._cull_hi):
            candidate |= np.all((flat >= lo) & (flat <= hi), axis=1)

        dist = np.full(len(flat), self.aabb_pad, dtype=float)
        if candidate.any():
            d, _ = self.tree.query(flat[candidate])
            dist[candidate] = d
        return dist.reshape(original_shape)


def build_wall_index(directory, exclude=(), target_spacing=2.0, seed=0,
                      aabb_pad=15.0, cull=True, **kwargs):
    """
    Convenience: load the electrode meshes once, merge them for the
    KDTree, and (by default) also pass their individual bounding boxes
    through for the AABB broad-phase filter (see WallIndex). Set cull=False
    to skip the pre-filter (e.g. for accuracy-checking against exact
    queries, where the sentinel distance would get in the way).

    kwargs are forwarded to load_electrode_meshes (pattern, n_electrodes, ...).
    """
    meshes = load_electrode_meshes(directory, **kwargs)
    kept = {num: m for num, m in meshes.items() if num not in exclude}
    if not kept:
        raise ValueError("exclude removed every electrode -- nothing left to merge")
    mesh = trimesh.util.concatenate(list(kept.values()))
    electrode_boxes = [m.bounds for m in kept.values()] if cull else None
    return WallIndex(mesh, target_spacing=target_spacing, seed=seed,
                      electrode_boxes=electrode_boxes, aabb_pad=aabb_pad)


if __name__ == "__main__":
    import time

    here = Path(__file__).resolve().parent
    # 1, 2, 19: source (particles start there), pipe/housing (bbox spans the
    # whole chamber, redundant with the outer-bbox chamber-exit check), and
    # Detector (landing there is success, not a wall hit).
    wall_index = build_wall_index(here, exclude=(1, 2, 19), target_spacing=2.0)
    print(f"WallIndex: {len(wall_index.points)} sample points, "
          f"AABB cull {'ON' if wall_index._cull_lo is not None else 'OFF'}")

    test_points = np.array([
        [395.0, 75.0, 77.0],   # beam start, mid-chamber -- should be far-ish
        [70.0, 75.0, 190.0],   # somewhere inside electrode_9's footprint
    ])
    for p, d in zip(test_points, wall_index.distance(test_points)):
        print(f"  point {p} -> approx distance to nearest wall surface: {d:.2f} mm")

    rng = np.random.default_rng(0)
    pts = rng.uniform([0, 0, 0], [484, 153, 484], size=(500_000, 3))
    t0 = time.time()
    wall_index.distance(pts)
    print(f"500k random-point query: {time.time()-t0:.2f}s "
          f"(2.5M-point worst case would extrapolate to ~{(time.time()-t0)*5:.1f}s)")
