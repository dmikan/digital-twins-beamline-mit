"""
physics.py
==========

Unified physics module for the Gemelo Digital.
Combines field grid representation (previously dual_grid.py), RK4 integration
(previously RK4_sim_basis.py and RK4_sim_basis_batch.py), and collision geometry
(previously electrode_geometry.py).

The whole of RK4 now works in 1mm resolution (same as SIMION).
"""

import collections
import pathlib
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
from scipy.interpolate import RegularGridInterpolator

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE

# Rigid-body transform from STL export space to SIMION workbench coordinates
STL_TO_WORKBENCH_ROTATION = np.array([
    [ 0.99738233, -0.06972016, -0.01770672],
    [ 0.07079483,  0.99522571,  0.06696917],
    [ 0.01297262, -0.06805272,  0.99760373],
])
STL_TO_WORKBENCH_TRANSLATION = np.array([311.72967676, 164.97727664, 175.84421467])

E_VMM_TO_ACCEL_MM = 1.0e6
EXCLUIR_DEFAULT = (1, 2, 19)

class IonSpecies:
    def __init__(self, mass, charge):
        self.mass = mass
        self.charge = charge

class ParticleState:
    def __init__(self, position, velocity):
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)

class Beam:
    def __init__(self, species, position, velocity):
        self.species = species
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)

        if self.position.ndim == 1:
            self.position = self.position[None, :]
            self.velocity = self.velocity[None, :]

        self.n_particles = self.position.shape[0]
        self.alive = np.ones(self.n_particles, dtype=bool)
        self.state = ParticleState(position=self.position, velocity=self.velocity)

class Trajectory:
    def __init__(self, beam):
        self.species = beam.species
        self.initial_state = beam.state
        self.states = [self.initial_state]

    def add_state(self, state):
        self.states.append(state)

class BatchTrajectory:
    def __init__(self, beam, config_indices):
        self.species = beam.species
        self.initial_state = beam.state
        self.states = [self.initial_state]
        self.config_indices = np.asarray(config_indices, dtype=int)

    def add_state(self, state):
        self.states.append(state)

def make_batch_beam(species, start_positions, start_velocities, n_configs):
    n = start_positions.shape[0]
    all_positions = np.tile(start_positions, (n_configs, 1))
    all_velocities = np.tile(start_velocities, (n_configs, 1))
    config_indices = np.repeat(np.arange(n_configs), n)
    beam = Beam(species=species, position=all_positions, velocity=all_velocities)
    return beam, config_indices

def split_batch_trajectory(batch_trajectory, config_indices, n_configs):
    config_indices = np.asarray(config_indices)
    from types import SimpleNamespace
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

# ----------------------------------------------------------------------
# Geometry and Wall Mesh / Distance Check
# ----------------------------------------------------------------------
def _stl_to_workbench_matrix(rotation=STL_TO_WORKBENCH_ROTATION, translation=STL_TO_WORKBENCH_TRANSLATION):
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T

def load_electrode_meshes(directory, pattern="electrode_{i}.stl", n_electrodes=19, start_index=1,
                            rotation=STL_TO_WORKBENCH_ROTATION, translation=STL_TO_WORKBENCH_TRANSLATION):
    directory = pathlib.Path(directory)
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
    meshes = load_electrode_meshes(directory, **kwargs)
    kept = [m for num, m in meshes.items() if num not in exclude]
    if not kept:
        raise ValueError("exclude removed every electrode -- nothing left to merge")
    return trimesh.util.concatenate(kept)

class WallIndex:
    def __init__(self, mesh, target_spacing=2.0, seed=0, electrode_boxes=None, aabb_pad=15.0):
        self.mesh = mesh
        self.target_spacing = target_spacing
        n_samples = max(1, int(mesh.area / target_spacing**2))
        surface_pts, _ = trimesh.sample.sample_surface(mesh, n_samples, seed=seed)
        points = np.vstack([surface_pts, mesh.vertices])
        self.points = points
        self.tree = cKDTree(points)
        self.aabb_pad = aabb_pad
        if electrode_boxes:
            boxes = np.asarray(electrode_boxes)
            self._cull_lo = boxes[:, 0, :] - aabb_pad
            self._cull_hi = boxes[:, 1, :] + aabb_pad
        else:
            self._cull_lo = self._cull_hi = None

    def distance(self, points):
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
    meshes = load_electrode_meshes(directory, **kwargs)
    kept = {num: m for num, m in meshes.items() if num not in exclude}
    if not kept:
        raise ValueError("exclude removed every electrode -- nothing left to merge")
    mesh = trimesh.util.concatenate(list(kept.values()))
    electrode_boxes = [m.bounds for m in kept.values()] if cull else None
    return WallIndex(mesh, target_spacing=target_spacing, seed=seed,
                      electrode_boxes=electrode_boxes, aabb_pad=aabb_pad)

class ParedesPA:
    def __init__(self, puntos_metal, target_spacing=2.0):
        self.puntos = np.asarray(puntos_metal, dtype=float)
        self.target_spacing = target_spacing
        self._tree = cKDTree(self.puntos)

    @classmethod
    def desde_proyecto(cls, root=ROOT, excluir=EXCLUIR_DEFAULT, verbose=True, incluir_corredores=False):
        root = pathlib.Path(root)
        # PREFERIDO: mascaras de pared ya pre-excluidas por IDENTIDAD del
        # PA (playpen/extract_wall_masks.lua): los electrodos a conservar
        # se ponen a 1 y los excluidos (1/2/19) a 0, y el voxel es pared
        # sii is_electrode & potential>0.5. Sin STL, sin clasificacion --
        # la STL esta desalineada ~25mm en el quad (medido 2026-07-06),
        # estas mascaras salen del propio PA igual que el campo.
        walls = [root / "basis_quad" / "mask_walls_global_2mm.csv",
                 root / "basis_quad" / "mask_walls_quad_1mm.csv"]
        if all(w.exists() for w in walls):
            pts = np.vstack([np.loadtxt(w, delimiter=",", skiprows=1) for w in walls])
            if verbose:
                print(f"[ParedesPA] {len(pts)} puntos de pared (PA pre-excluido, sin STL)")
            return cls(pts)

        # FALLBACK legacy: metal completo + clasificacion STL para excluir.
        archivos = [root / "basis_quad" / "mask_global_2mm.csv",
                    root / "basis_quad" / "mask_quad_1mm.csv"]
        if incluir_corredores:
            for extra in ("mask_c1_1mm.csv", "mask_c2_1mm.csv"):
                p = root / "basis_quad" / extra
                if p.exists():
                    archivos.append(p)
        pts = np.vstack([np.loadtxt(f, delimiter=",", skiprows=1) for f in archivos])
        if verbose:
            print(f"[ParedesPA] {len(pts)} puntos de metal (fallback STL)")

        if excluir:
            meshes = load_electrode_meshes(root)
            arboles = {}
            for e, mesh in meshes.items():
                muestras, _ = mesh.sample(max(200, int(mesh.area / 25.0)), return_index=True)
                arboles[e] = cKDTree(muestras)
            dist_por_e = np.full((len(meshes), len(pts)), np.inf)
            orden_e = sorted(meshes)
            for j, e in enumerate(orden_e):
                dist_por_e[j], _ = arboles[e].query(pts, workers=-1)
            duenio = np.array(orden_e)[np.argmin(dist_por_e, axis=0)]
            keep = ~np.isin(duenio, list(excluir))
            if verbose:
                for e in excluir:
                    print(f"[ParedesPA]   excluido electrodo {e}: {int((duenio == e).sum())} puntos")
            pts = pts[keep]
        if verbose:
            print(f"[ParedesPA] indice final: {len(pts)} puntos de metal")
        return cls(pts)

    def distance(self, points):
        d, _ = self._tree.query(np.asarray(points, dtype=float), workers=-1)
        return d

# ----------------------------------------------------------------------
# Field Maps
# ----------------------------------------------------------------------
class BasisFieldMap:
    def __init__(self, basis_files):
        self.basis_files = [pathlib.Path(f) for f in basis_files]
        self.n_electrodes = len(self.basis_files)
        if self.n_electrodes == 0:
            raise ValueError("basis_files is empty")
        self.x = self.y = self.z = None
        self.Nx = self.Ny = self.Nz = None
        self.dx = self.dy = self.dz = None
        self.coords = None
        self.basis_V = None
        self.voltages = np.zeros(self.n_electrodes)
        self.V = None
        self.E = None
        self._interpolator = None
        self._load_basis_files()
        self.set_voltages(np.zeros(self.n_electrodes))

    @classmethod
    def from_directory(cls, directory, pattern="basis_electrode_{i}.csv", n_electrodes=19, start_index=1):
        directory = pathlib.Path(directory)
        files = []
        for i in range(start_index, start_index + n_electrodes):
            fpath = directory / pattern.format(i=i)
            if not fpath.exists():
                raise FileNotFoundError(f"Expected basis file not found: {fpath}")
            files.append(fpath)
        return cls(files)

    def _load_basis_files(self):
        for idx, fpath in enumerate(self.basis_files):
            data = np.loadtxt(fpath, delimiter=",", skiprows=1)
            order = np.lexsort((data[:, 2], data[:, 1], data[:, 0]))
            data = data[order]
            x = np.unique(data[:, 0])
            y = np.unique(data[:, 1])
            z = np.unique(data[:, 2])
            Nx, Ny, Nz = len(x), len(y), len(z)

            expected_points = Nx * Ny * Nz
            if len(data) != expected_points:
                raise ValueError(f"{fpath}: expected {expected_points} grid points, got {len(data)}.")

            V_i = data[:, 3].reshape(Nx, Ny, Nz)
            if idx == 0:
                self.x, self.y, self.z = x, y, z
                self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
                self.basis_V = np.empty((self.n_electrodes, Nx, Ny, Nz), dtype=float)
            else:
                if not (np.array_equal(x, self.x) and np.array_equal(y, self.y) and np.array_equal(z, self.z)):
                    raise ValueError(f"{fpath}: grid does not match the reference grid.")
            self.basis_V[idx] = V_i

        self.dx = self.x[1] - self.x[0] if self.Nx > 1 else 1.0
        self.dy = self.y[1] - self.y[0] if self.Ny > 1 else 1.0
        self.dz = self.z[1] - self.z[0] if self.Nz > 1 else 1.0
        self.coords = np.stack(np.meshgrid(self.x, self.y, self.z, indexing="ij"), axis=-1)

    def set_voltages(self, voltages):
        voltages = np.asarray(voltages, dtype=float)
        if voltages.shape != (self.n_electrodes,):
            raise ValueError(f"voltages must have shape ({self.n_electrodes},), got {voltages.shape}")
        self.voltages = voltages
        self.V = np.tensordot(voltages, self.basis_V, axes=(0, 0))
        self._compute_field_from_potential()
        self._build_interpolator()
        return self.V

    def _compute_field_from_potential(self):
        dVdx, dVdy, dVdz = np.gradient(self.V, self.dx, self.dy, self.dz)
        self.E = np.stack((-dVdx, -dVdy, -dVdz), axis=-1)

    def _build_interpolator(self):
        self._interpolator = RegularGridInterpolator(
            points=(self.x, self.y, self.z),
            values=self.E,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

    def field(self, positions):
        positions = np.asarray(positions)
        if positions.shape[-1] != 3:
            raise ValueError("positions must have shape (..., 3)")
        original_shape = positions.shape[:-1]
        positions = positions.reshape(-1, 3)
        clipped = np.array(positions, copy=True)
        clipped[:, 0] = np.clip(clipped[:, 0], self.x.min(), self.x.max())
        clipped[:, 1] = np.clip(clipped[:, 1], self.y.min(), self.y.max())
        clipped[:, 2] = np.clip(clipped[:, 2], self.z.min(), self.z.max())
        E = self._interpolator(clipped)
        outside = (
            (positions[:, 0] < self.x.min()) | (positions[:, 0] > self.x.max()) |
            (positions[:, 1] < self.y.min()) | (positions[:, 1] > self.y.max()) |
            (positions[:, 2] < self.z.min()) | (positions[:, 2] > self.z.max())
        )
        if np.any(outside):
            E[outside] = 0.0
        return E.reshape(*original_shape, 3)

class BatchBasisFieldMap(BasisFieldMap):
    def field(self, positions):
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
        voltages_batch = np.asarray(voltages_batch, dtype=dtype)
        if voltages_batch.ndim != 2 or voltages_batch.shape[1] != self.n_electrodes:
            raise ValueError(f"voltages_batch must have shape (M, {self.n_electrodes})")
        self.M = voltages_batch.shape[0]
        self.voltages_batch = voltages_batch
        basis_V = self.basis_V.astype(dtype, copy=False)
        self.V_batch = np.tensordot(voltages_batch, basis_V, axes=([1], [0]))
        dVdx, dVdy, dVdz = np.gradient(self.V_batch, self.dx, self.dy, self.dz, axis=(1, 2, 3))
        self.E_batch = -np.stack((dVdx, dVdy, dVdz), axis=-1).astype(dtype, copy=False)
        return self.E_batch

    def field_batch(self, config_indices, positions):
        config_indices = np.asarray(config_indices)
        positions = np.asarray(positions)
        if positions.shape[0] != config_indices.shape[0]:
            raise ValueError("config_indices and positions must have the same length")

        clipped = np.array(positions, copy=True)
        clipped[:, 0] = np.clip(clipped[:, 0], self.x.min(), self.x.max())
        clipped[:, 1] = np.clip(clipped[:, 1], self.y.min(), self.y.max())
        clipped[:, 2] = np.clip(clipped[:, 2], self.z.min(), self.z.max())

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

class _MapaArrays(BatchBasisFieldMap):
    def __init__(self, x, y, z, basis_V):
        self.basis_files = []
        self.n_electrodes = basis_V.shape[0]
        self.x, self.y, self.z = x, y, z
        self.Nx, self.Ny, self.Nz = len(x), len(y), len(z)
        self.basis_V = basis_V
        self.dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
        self.dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
        self.dz = float(z[1] - z[0]) if len(z) > 1 else 1.0
        self.coords = None
        self.voltages = np.zeros(self.n_electrodes)
        self.V = None
        self.E = None
        self._interpolator = None
        self.set_voltages(np.zeros(self.n_electrodes))

def _cargar_caja(root, nombre, archivos):
    cache = root / "basis_quad" / f"cache_{nombre}.npz"
    if cache.exists() and all(cache.stat().st_mtime >= f.stat().st_mtime for f in archivos):
        d = np.load(cache)
        return _MapaArrays(d["x"], d["y"], d["z"], d["basis_V"])

    try:
        import pandas as pd
        leer = lambda f: pd.read_csv(f).to_numpy(dtype=np.float64)
    except ImportError:
        leer = lambda f: np.loadtxt(f, delimiter=",", skiprows=1)

    basis_V = X = Y = Z = None
    for idx, f in enumerate(archivos):
        data = leer(f)
        data = data[np.lexsort((data[:, 2], data[:, 1], data[:, 0]))]
        x, y, z = (np.unique(data[:, c]) for c in (0, 1, 2))
        if basis_V is None:
            X, Y, Z = x, y, z
            basis_V = np.empty((len(archivos), len(x), len(y), len(z)), dtype=np.float32)
        basis_V[idx] = data[:, 3].reshape(len(x), len(y), len(z)).astype(np.float32)
    np.savez(cache, x=X, y=Y, z=Z, basis_V=basis_V)
    print(f"[CampoFino] cache construido: {cache.name} ({basis_V.nbytes / 1e6:.0f}MB)")
    return _MapaArrays(X, Y, Z, basis_V)

class CampoFino(BatchBasisFieldMap):
    """
    Field map with fine boxes (quad, c1, c2) over the beam corridor plus a
    coarse global map; queries are routed to the finest box that contains
    them. Boxes are extracted at 2.5mm (revertido 2026-07-06: el barrido
    controlado playpen/comparar_resolucion.py mostro ranking identico a
    1mm -- +0.168 vs +0.160 -- por ~15x menos puntos y memoria; el 1mm
    solo afinaba live/dead marginalmente, no el ranking).
    """
    def __init__(self, basis_files, cajas_finas, root=ROOT):
        super().__init__(basis_files)
        self.finos = []
        for nombre, archivos in cajas_finas:
            mapa = _cargar_caja(pathlib.Path(root), nombre, archivos)
            lo = np.array([mapa.x.min(), mapa.y.min(), mapa.z.min()])
            hi = np.array([mapa.x.max(), mapa.y.max(), mapa.z.max()])
            self.finos.append((nombre, mapa, lo, hi))
            print(f"[CampoFino] caja '{nombre}': {mapa.Nx}x{mapa.Ny}x{mapa.Nz} ({mapa.dx:.1f}mm) x[{lo[0]:g},{hi[0]:g}]")

    @classmethod
    def desde_proyecto(cls, root=ROOT, n_electrodes=19):
        root = pathlib.Path(root)
        gruesos = [root / f"basis_electrode_{i}.csv" for i in range(1, n_electrodes + 1)]
        cajas = []
        for nombre, patron in (("quad", "basis_quad_electrode_{i}.csv"),
                               ("c1", "basis_c1_electrode_{i}.csv"),
                               ("c2", "basis_c2_electrode_{i}.csv")):
            archivos = [root / "basis_quad" / patron.format(i=i) for i in range(1, n_electrodes + 1)]
            if all(f.exists() for f in archivos):
                cajas.append((nombre, archivos))
        return cls(gruesos, cajas, root=root)

    def set_voltages_batch(self, voltages_batch, dtype=np.float64):
        out = super().set_voltages_batch(voltages_batch, dtype=dtype)
        for _, mapa, _, _ in self.finos:
            mapa.set_voltages_batch(voltages_batch, dtype=np.float32)
        return out

    # Ruteo caja-primero: se asigna cada punto a su caja fina y el mapa
    # grueso se evalua SOLO sobre los sobrantes (fuera de toda caja), en
    # vez de calcular el grueso para TODOS y sobrescribir los de adentro.
    # Como el corredor del haz esta casi todo cubierto por cajas finas,
    # antes se tiraba la mayor parte del trabajo grueso. Resultado
    # IDENTICO (los de adentro ya recibian el valor fino; los de afuera,
    # el grueso) -- solo se evita computar lo que se iba a descartar.
    def field_batch(self, config_indices, positions):
        p = np.asarray(positions)
        ci = np.asarray(config_indices)
        E = np.empty((len(p), 3))
        pendiente = np.ones(len(p), dtype=bool)
        for _, mapa, lo, hi in self.finos:
            dentro = pendiente & np.all((p >= lo) & (p <= hi), axis=-1)
            if dentro.any():
                E[dentro] = mapa.field_batch(ci[dentro], p[dentro])
                pendiente &= ~dentro
        if pendiente.any():
            E[pendiente] = super().field_batch(ci[pendiente], p[pendiente])
        return E

    def field(self, positions):
        p = np.asarray(positions)
        forma = p.shape[:-1]
        plano = p.reshape(-1, 3)
        E = np.empty((len(plano), 3))
        pendiente = np.ones(len(plano), dtype=bool)
        for _, mapa, lo, hi in self.finos:
            dentro = pendiente & np.all((plano >= lo) & (plano <= hi), axis=-1)
            if dentro.any():
                E[dentro] = mapa.field(plano[dentro])
                pendiente &= ~dentro
        if pendiente.any():
            E[pendiente] = super().field(plano[pendiente])
        return E.reshape(*forma, 3)

# ----------------------------------------------------------------------
# RK4 Integrator
# ----------------------------------------------------------------------
class BatchRK4Integrator:
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

# ----------------------------------------------------------------------
# Physics Loader
# ----------------------------------------------------------------------
Fisica = collections.namedtuple("Fisica", "bfm wall margen chunk_screening dual")
# El margen (0.5mm) es del MURO (ParedesPA, metal del PA) -- no depende de
# la resolucion del campo. El chunk sube a 32 con cajas de 2.5mm: pesan
# ~15x menos que las de 1mm, asi que caben mas configs por chunk (menos
# overhead de Python por paso de integracion).
MARGEN_FINO = 0.5
MARGEN_STL = 1.5
CHUNK_FINO = 32
CHUNK_STL = 50

def cargar_fisica(root=ROOT, n_electrodes=19, verbose=True):
    root = pathlib.Path(root)
    bq = root / "basis_quad"
    tiene_fino = (bq / "mask_global_2mm.csv").exists() and all(
        (bq / f"basis_{caja}_electrode_{i}.csv").exists()
        for caja in ("quad", "c1", "c2") for i in (1, n_electrodes))
    if tiene_fino:
        if verbose:
            print("[physics] cajas finas 2.5mm (CampoFino + ParedesPA, margen 0.5)")
        bfm = CampoFino.desde_proyecto(root, n_electrodes=n_electrodes)
        wall = ParedesPA.desde_proyecto(root, verbose=verbose)
        return Fisica(bfm, wall, MARGEN_FINO, CHUNK_FINO, True)

    if verbose:
        print("[physics] fallback clasica (bases 50^3 + STL, margen 1.5)")
    bfm = BatchBasisFieldMap.from_directory(root, n_electrodes=n_electrodes)
    wall = build_wall_index(root, exclude=EXCLUIR_DEFAULT, target_spacing=2.0)
    return Fisica(bfm, wall, MARGEN_STL, CHUNK_STL, False)
