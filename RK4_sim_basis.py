import numpy as np
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
import optimizer as op

class BasisFieldMap:
    def __init__(self, basis_files):
        """
        Parameters
        ----------
        basis_files : list of str or Path
            Ordered list of basis CSV files, one per electrode
            (basis_files[0] = electrode 1, basis_files[1] = electrode 2, ...).
            Each file must have columns x,y,z,V and share the SAME grid.
        """
        self.basis_files = [Path(f) for f in basis_files]
        self.n_electrodes = len(self.basis_files)

        if self.n_electrodes == 0:
            raise ValueError("basis_files is empty")

        self.x = self.y = self.z = None
        self.Nx = self.Ny = self.Nz = None
        self.dx = self.dy = self.dz = None
        self.coords = None  # shape (Nx, Ny, Nz, 3)

        # basis_V: shape (n_electrodes, Nx, Ny, Nz)
        self.basis_V = None

        # Current combined state (populated by set_voltages)
        self.voltages = np.zeros(self.n_electrodes)
        self.V = None                 # combined potential, shape (Nx, Ny, Nz)
        self.E = None                 # combined field, shape (Nx, Ny, Nz, 3)
        self._interpolator = None

        self._load_basis_files()
        # Initialize with all-zero voltages so the object is immediately usable.
        self.set_voltages(np.zeros(self.n_electrodes))

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_directory(cls, directory, pattern="basis_electrode_{i}.csv", n_electrodes=19,
                        start_index=1):
        """
        Convenience constructor: build the ordered file list from a
        directory + naming pattern instead of listing files by hand.
        """
        pattern = str(pattern)
        directory = Path(directory)
        files = []
        for i in range(start_index, n_electrodes+ start_index):
            fpath = directory / pattern.format(i=i)
            if not fpath.exists():
                raise FileNotFoundError(f"Expected basis file not found: {fpath}")
            files.append(fpath)
        return cls(files)

    # ------------------------------------------------------------------
    # Loading (done once)
    # ------------------------------------------------------------------
    def _load_basis_files(self):
        for idx, fpath in enumerate(self.basis_files):
            data = np.loadtxt(fpath, delimiter=",", skiprows=1)

            # Sort into canonical (x, then y, then z) order
            order = np.lexsort((data[:, 2], data[:, 1], data[:, 0]))
            data = data[order]

            x = np.unique(data[:, 0])
            y = np.unique(data[:, 1])
            z = np.unique(data[:, 2])
            Nx, Ny, Nz = len(x), len(y), len(z)

            expected_points = Nx * Ny * Nz
            if len(data) != expected_points:
                raise ValueError(
                    f"{fpath}: expected {expected_points} grid points, got {len(data)}. "
                    "Check CSV ordering/completeness."
                )

            V_i = data[:, 3].reshape(Nx, Ny, Nz)

            if idx == 0:
                # First file defines the reference grid
                self.x, self.y, self.z = x, y, z
                self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
                self.basis_V = np.empty((self.n_electrodes, Nx, Ny, Nz), dtype=float)
            else:
                # All subsequent files must share the exact same grid
                if not (np.array_equal(x, self.x) and np.array_equal(y, self.y)
                        and np.array_equal(z, self.z)):
                    raise ValueError(
                        f"{fpath}: grid does not match the grid from "
                        f"{self.basis_files[0]}. All basis files must share one grid."
                    )

            self.basis_V[idx] = V_i

        # Grid spacing (assumes uniform spacing, as produced by extract_PA.lua)
        self.dx = self.x[1] - self.x[0] if self.Nx > 1 else 1.0
        self.dy = self.y[1] - self.y[0] if self.Ny > 1 else 1.0
        self.dz = self.z[1] - self.z[0] if self.Nz > 1 else 1.0

        # Create the coordinate grid
        self.coords = np.stack(
            np.meshgrid(self.x, self.y, self.z, indexing="ij"),
            axis=-1
        )

    # ------------------------------------------------------------------
    # Updating voltages (cheap -- no file I/O, just a linear combination)
    # ------------------------------------------------------------------
    def set_voltages(self, voltages):
        """
        Recombine the basis potentials with a new voltage vector and
        recompute the electric field. This is the method you call every
        time you want to try a new electrode configuration.

        Parameters
        ----------
        voltages : array_like, shape (n_electrodes,)
            voltages[i] is the voltage applied to electrode i+1.
        """
        voltages = np.asarray(voltages, dtype=float)
        if voltages.shape != (self.n_electrodes,):
            raise ValueError(
                f"voltages must have shape ({self.n_electrodes},), got {voltages.shape}"
            )
        self.voltages = voltages

        # V(x,y,z) = sum_i voltages[i] * basis_V[i](x,y,z)
        self.V = np.tensordot(voltages, self.basis_V, axes=(0, 0))  # (Nx, Ny, Nz)

        self._compute_field_from_potential()
        self._build_interpolator()
        return self.V

    def update_voltage(self, electrode_index, voltage):
        """
        Convenience method to change a single electrode's voltage
        (1-based electrode_index) and recombine.
        """
        new_voltages = self.voltages.copy()
        new_voltages[electrode_index - 1] = voltage
        return self.set_voltages(new_voltages)

    # ------------------------------------------------------------------
    # Field computation
    # ------------------------------------------------------------------
    def _compute_field_from_potential(self):
        """E = -grad(V), computed on the grid via central differences."""
        dVdx, dVdy, dVdz = np.gradient(self.V, self.dx, self.dy, self.dz)
        Ex = -dVdx
        Ey = -dVdy
        Ez = -dVdz
        self.E = np.stack((Ex, Ey, Ez), axis=-1)  # (Nx, Ny, Nz, 3)

    def _build_interpolator(self):
        self._interpolator = RegularGridInterpolator(
            points=(self.x, self.y, self.z),
            values=self.E,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

    # ------------------------------------------------------------------
    # Querying the field (same interface as FieldMap.field in RK4_sim_claude.py)
    # ------------------------------------------------------------------
    def field(self, positions):
        """
        Evaluate the electric field at one or many positions, for the
        CURRENT voltage configuration (whatever was last passed to
        set_voltages / update_voltage).

        Parameters
        ----------
        positions : (..., 3) array_like
            Last dimension contains [x, y, z].

        Returns
        -------
        ndarray, same leading shape as positions, with a trailing (3,)
        """
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

    def potential(self, positions):
        """
        Evaluate the scalar potential V at one or many positions, for the
        current voltage configuration. Uses simple linear interpolation
        on the same grid.
        """
        positions = np.asarray(positions)
        original_shape = positions.shape[:-1]
        positions = positions.reshape(-1, 3)

        interp = RegularGridInterpolator(
            points=(self.x, self.y, self.z),
            values=self.V,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        V = interp(positions)
        return V.reshape(*original_shape)

    # ------------------------------------------------------------------
    def summary(self):
        print("========== BASIS FIELD MAP ==========")
        print(f"Electrodes  : {self.n_electrodes}")
        print(f"Grid shape  : ({self.Nx}, {self.Ny}, {self.Nz})")
        print(f"x : {self.x.min():.3f} -> {self.x.max():.3f}  (dx={self.dx:.4f})")
        print(f"y : {self.y.min():.3f} -> {self.y.max():.3f}  (dy={self.dy:.4f})")
        print(f"z : {self.z.min():.3f} -> {self.z.max():.3f}  (dz={self.dz:.4f})")
        print(f"Current voltages: {self.voltages}")
        if self.V is not None:
            print(f"Combined V range: {self.V.min():.4f} -> {self.V.max():.4f}")


def random_voltage_demo(bfm, n_samples=2, voltage_range=(-1000.0, 1000.0)):
    """Demonstrate how random voltage vectors change the combined field map."""
    print("\n========== RANDOM VOLTAGE DEMO ==========")
    for sample in range(1, n_samples + 1):
        voltages = np.random.uniform(voltage_range[0], voltage_range[1], size=bfm.n_electrodes)
        bfm.set_voltages(voltages)

        print(f"\nSample {sample}: random voltages")
        print(voltages)
        print("Combined potential range:", bfm.V.min(), "to", bfm.V.max())
        print("Combined field ranges:")
        print("  Ex:", bfm.E[..., 0].min(), "to", bfm.E[..., 0].max())
        print("  Ey:", bfm.E[..., 1].min(), "to", bfm.E[..., 1].max())
        print("  Ez:", bfm.E[..., 2].min(), "to", bfm.E[..., 2].max())
        center = np.array([[bfm.x.mean(), bfm.y.mean(), bfm.z.mean()]])
        print("Field at grid center:", bfm.field(center))

class FieldMap:

    def __init__(self, filename):

        self.filename = Path(filename)

        # Load CSV
        data = np.loadtxt(
            self.filename,
            delimiter=",",
            skiprows=1
        )

        # Sort data (x, then y, then z)
        order = np.lexsort((data[:, 2], data[:, 1], data[:, 0]))
        data = data[order]

        # Coordinate vectors
        self.x = np.unique(data[:, 0])
        self.y = np.unique(data[:, 1])
        self.z = np.unique(data[:, 2])

        self.Nx = len(self.x)
        self.Ny = len(self.y)
        self.Nz = len(self.z)

        expected_points = self.Nx * self.Ny * self.Nz
        if len(data) != expected_points:
            raise ValueError(
                f"Expected {expected_points} grid points, but loaded {len(data)}. "
                "Check the CSV ordering or grid completeness."
            )

        # Coordinate grid: shape (Nx, Ny, Nz, 3)
        self.coords = np.stack(
            np.meshgrid(self.x, self.y, self.z, indexing="ij"),
            axis=-1
        )

        # Electric field arrays: shape (Nx, Ny, Nz)
        self.Ex = data[:, 3].reshape(self.Nx, self.Ny, self.Nz)
        self.Ey = data[:, 4].reshape(self.Nx, self.Ny, self.Nz)
        self.Ez = data[:, 5].reshape(self.Nx, self.Ny, self.Nz)

        # Field vector at each grid point: shape (Nx, Ny, Nz, 3)
        self.E = np.stack((self.Ex, self.Ey, self.Ez), axis=-1)

        # Combined coordinate + field array: shape (Nx, Ny, Nz, 6)
        self.coord_field = np.concatenate((self.coords, self.E), axis=-1)
        self.interpolator = RegularGridInterpolator(
                                points=(self.x, self.y, self.z),
                                values=self.E,
                                method="linear",
                                bounds_error=False,
                                fill_value=np.nan
                            )


    def summary(self):

        print("========== FIELD MAP ==========")
        print(f"Grid shape : ({self.Nx}, {self.Ny}, {self.Nz})")
        print()

        print(f"x : {self.x.min()} -> {self.x.max()}")
        print(f"y : {self.y.min()} -> {self.y.max()}")
        print(f"z : {self.z.min()} -> {self.z.max()}")
        print()

        print(f"Ex : {self.Ex.min()} -> {self.Ex.max()}")
        print(f"Ey : {self.Ey.min()} -> {self.Ey.max()}")
        print(f"Ez : {self.Ez.min()} -> {self.Ez.max()}")

    def field(self, positions):
        """
        Evaluate the electric field at one or many positions.

        Parameters
        ----------
        positions : (..., 3) array_like
            Last dimension contains [x, y, z].

        Returns
        -------
        ndarray
            Electric field with the same leading shape as positions.
        """
        positions = np.asarray(positions)

        if positions.shape[-1] != 3:
            raise ValueError("positions must have shape (..., 3)")

        original_shape = positions.shape[:-1]

        # Flatten into (N, 3)
        positions = positions.reshape(-1, 3)

        # Clamp positions to the valid interpolation domain so the particle can
        # move slightly outside the sampled field box without crashing.
        clipped = np.array(positions, copy=True)
        clipped[:, 0] = np.clip(clipped[:, 0], self.x.min(), self.x.max())
        clipped[:, 1] = np.clip(clipped[:, 1], self.y.min(), self.y.max())
        clipped[:, 2] = np.clip(clipped[:, 2], self.z.min(), self.z.max())

        # Interpolate
        E = self.interpolator(clipped)

        # Outside the sampled box, return zero field instead of raising.
        outside = (
            (positions[:, 0] < self.x.min()) | (positions[:, 0] > self.x.max()) |
            (positions[:, 1] < self.y.min()) | (positions[:, 1] > self.y.max()) |
            (positions[:, 2] < self.z.min()) | (positions[:, 2] > self.z.max())
        )
        if np.any(outside):
            E[outside] = 0.0

        # Restore original shape
        return E.reshape(*original_shape, 3)
    
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

        self.state = ParticleState(
            position=self.position,
            velocity=self.velocity,
        )

class Trajectory:
    def __init__(self, beam):
        self.species = beam.species
        self.initial_state = beam.state
        self.states = [self.initial_state]  # List of ParticleState instances

    def add_state(self, state):
        self.states.append(state)

class RK4Integrator:
    def __init__(self, field_map):
        self.field_map = field_map

    def integrate(self, trajectory, dt, num_steps):
        """
        Integrate the trajectory using RK4 method.

        Parameters
        ----------
        trajectory : Trajectory
            The trajectory to integrate.
        dt : float
            Time step for integration.
        num_steps : int
            Number of integration steps.
        """
        state = trajectory.initial_state
        species = trajectory.species  # captured once; _derivative has no access to `trajectory`
        for _ in range(num_steps):
            k1 = self._derivative(state, species)
            k2 = self._derivative(ParticleState(state.position + 0.5 * dt * k1[0],
                                                state.velocity + 0.5 * dt * k1[1]), species)
            k3 = self._derivative(ParticleState(state.position + 0.5 * dt * k2[0],
                                                state.velocity + 0.5 * dt * k2[1]), species)
            k4 = self._derivative(ParticleState(state.position + dt * k3[0],
                                                state.velocity + dt * k3[1]), species)

            new_position = state.position + (dt / 6) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
            new_velocity = state.velocity + (dt / 6) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])

            state = ParticleState(new_position, new_velocity)
            trajectory.add_state(state)

    def _derivative(self, state, species):
        """
        Compute the derivative of the particle's state.

        Parameters
        ----------
        state : ParticleState
            The current state of the particle.
        species : IonSPecies
            Mass and charge of the particle being integrated.

        Returns
        -------
        tuple
            A tuple containing the derivative of position and velocity.
        """
        E_field = self.field_map.field(state.position)
        acceleration = species.charge * E_field / species.mass
        return (state.velocity, acceleration)

# First attempt at loading potential basis files and creating a field map. Adjust the path as needed.
if __name__ == "__main__":
    df = pd.read_csv("beamline_results.csv")
    X = df[
            "params_V3",
            "params_V6",
            "params_V9",
            "params_V10",
            "params_V11",
            "params_V12",
            "params_V15",
            "params_V18",
            ]
    directory = './'
    n_electrodes = 19  # Adjust this to the actual number of electrodes you have
    bfm = BasisFieldMap.from_directory(directory, n_electrodes=n_electrodes)
    bfm.set_voltages([500.0, 0.0, -895.921163747548, 0.0, 0.0, 689.815550200552, 0.0, 0.0 , -33.2956871479055, 778.366815914241, 331.740232885476, -352.050845144708, 0.0, 0.0, 376.220581070828, 0.0, 0.0, -123.329027787132, -2000])  # Initialize with zero voltages
    bfm.summary()

    fieldData = bfm  # Evaluate the field at all grid points

    # fieldData.summary()

    print("Coordinate grid shape:", fieldData.coords.shape)
    # print("Field vector shape:", fieldData.E.shape)
    # print("Combined coord+field shape:", fieldData.coord_field.shape)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # --- Experiment bounds in mm ---
    x_min, x_max = 0.0, 484.0
    y_min, y_max = 0.0, 153.0
    z_min, z_max = 0.0, 484.0

    print(f"Experiment size (mm): x={x_min:.1f}-{x_max:.1f}, y={y_min:.1f}-{y_max:.1f}, z={z_min:.1f}-{z_max:.1f}")

    integrator = RK4Integrator(fieldData)
    # --- Extra beam with N particles and a gaussian spread ---
    # -------------------------------
    # Beam parameters
    # -------------------------------
    N = 500

    u = 1.66053906660e-27      # kg
    e = 1.602176634e-19        # C

    mass = 28 * u
    charge = 1 * e

    species = IonSpecies(
        mass=mass,
        charge=charge
    )

    # -------------------------------
    # Initial position
    # -------------------------------
    start_positions = np.zeros((N,3))
    start_positions[:] = [395.0,75.0,77.0]      

    # -------------------------------
    # Energy distribution
    # -------------------------------
    mean_energy = 15.0      # eV
    std_energy = 0.42466    # eV

    energies = np.random.normal(
        mean_energy,
        std_energy,
        N
    )

    # Evitar energías negativas
    energies = np.clip(energies,0,None)

    # Convertir a Joules
    energies *= e

    # ============================================================
    # Angular cone (SIMION half-angle = 15°)
    # ============================================================
    half_angle = np.deg2rad(15)

    cos_theta = np.random.uniform(np.cos(half_angle), 1.0, N)
    theta = np.arccos(cos_theta)
    phi = np.random.uniform(0, 2*np.pi, N)

    directions = np.empty((N,3))

    # eje del cono = (-1,0,0)
    directions[:,0] = -np.cos(theta)
    directions[:,1] =  np.sin(theta)*np.cos(phi)
    directions[:,2] =  np.sin(theta)*np.sin(phi)

    # Normalizar por seguridad
    directions /= np.linalg.norm(directions, axis=1)[:,None]
    # -------------------------------
    # Initial velocities
    # -------------------------------
    speeds = 1000*np.sqrt(2*energies/mass) #quedan en mm/s

    start_velocities = np.zeros((N,3))
    start_velocities = speeds[:, None] * directions   # dirección (-1,0,0)
    dt = 5e-8
    num_steps = 5000
    import time
    start_time = time.time()
    beam_multi = Beam(
        species=species,
        position=start_positions,
        velocity=start_velocities,
    )
    multi_beam_traj = Trajectory(beam_multi)
    integrator.integrate(multi_beam_traj, dt, num_steps)
    fin = time.time()-start_time
    print(fin)

    positions_multi = np.array([
        state.position
        for state in multi_beam_traj.states
    ])
    disp = positions_multi[1] - positions_multi[0]

    print(disp[:5])
    print(np.linalg.norm(disp, axis=1)[:5])
    print((start_velocities * dt)[:5])
    # --- Downsampling stride, since plotting every grid point is usually unreadable ---
    stride = 4  # increase if the plot looks too dense/cluttered, decrease for more detail

    coords = fieldData.coords[::stride, ::stride, ::stride]
    E = fieldData.E[::stride, ::stride, ::stride]

    X, Y, Z = coords[..., 0], coords[..., 1], coords[..., 2]
    Ex, Ey, Ez = E[..., 0], E[..., 1], E[..., 2]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # normalize arrow length for readability; magnitude is still encoded via color
    E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

    quiver = ax.quiver(
        X, Y, Z, Ex, Ey, Ez,
        length=0.5 * min(x_max - x_min, y_max - y_min, z_max - z_min) / max(coords.shape[:3]),
        normalize=True,
        cmap="viridis",
    )

    # draw the bounding box wireframe
    def draw_box(ax, xmin, xmax, ymin, ymax, zmin, zmax, **kwargs):
        corners = np.array([
            [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
            [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
        ])
        edges = [
            (0,1),(1,2),(2,3),(3,0),  # bottom
            (4,5),(5,6),(6,7),(7,4),  # top
            (0,4),(1,5),(2,6),(3,7),  # verticals
        ]
        for i, j in edges:
            ax.plot(*zip(corners[i], corners[j]), **kwargs)

    draw_box(ax, x_min, x_max, y_min, y_max, z_min, z_max, color="red", lw=1, linestyle="--")

    # plot every path from the 100-particle beam
    for path in positions_multi:
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color="blue", lw=0.5, alpha=0.25)

    ax.scatter(positions_multi[:, 0, 0], positions_multi[:, 0, 1], positions_multi[:, 0, 2], color="royalblue", s=20, label="100-particle starts")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Electric Field Map with Particle Trajectory")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("field_map.png", dpi=150)
    plt.show()

# --- Tabular printout ---
#print(f"{'Step':>6} {'x':>12} {'y':>12} {'z':>12}")
#for i, pos in enumerate(positions_single):
    #print(f"{i:6d} {pos[0]:12.6e} {pos[1]:12.6e} {pos[2]:12.6e}")

# E_interp = np.zeros_like(fieldData.E)

# # Loop over every grid point
# for i in range(fieldData.coords.shape[0]):
#     for j in range(fieldData.coords.shape[1]):
#         for k in range(fieldData.coords.shape[2]):

#             position = fieldData.coords[i, j, k]

#             E_interp[i, j, k] = fieldData.field(position)

# print(E_interp.shape)
# # print(fieldData.coords)
# # print(fieldData.field([10.24,11.12,12.89]))  # Example position to test the field method

# # Difference between the original field and interpolated field
# difference = E_interp - fieldData.E

# print("Max Ex:", fieldData.E[:, :, :, 0].max())
# print("Max Ey:", fieldData.E[:, :, :, 1].max())
# print("Max Ez:", fieldData.E[:, :, :, 2].max())
# print("Max Ex:", E_interp[:, :, :, 0].max())
# print("Max Ey:", E_interp[:, :, :, 1].max())
# print("Max Ez:", E_interp[:, :, :, 2].max())

# # Component-wise absolute error
# abs_error = np.abs(difference)

# print("===== INTERPOLATION TEST =====")
# print(f"Maximum absolute error : {abs_error.max():.6e}")
# print(f"Mean absolute error    : {abs_error.mean():.6e}")
# print(f"RMS absolute error     : {np.sqrt(np.mean(abs_error**2)):.6e}")

# # Vector error magnitude at every grid point
# vector_error = np.linalg.norm(difference, axis=3)

# print()
# print("===== VECTOR ERROR =====")
# print(f"Maximum vector error : {vector_error.max():.6e}")
# print(f"Mean vector error    : {vector_error.mean():.6e}")
# print(f"RMS vector error     : {np.sqrt(np.mean(vector_error**2)):.6e}")

# # Check whether they're essentially identical
# print()
# print("Arrays equal? ", np.array_equal(E_interp, fieldData.E))
# print("Arrays close? ", np.allclose(E_interp, fieldData.E, atol=1e-12))