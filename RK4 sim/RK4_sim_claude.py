import numpy as np
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

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
        self.n_particles = self.position.shape[0] if self.position.ndim > 1 else 1
        self.alive = np.ones(self.n_particles, dtype=bool)

        if self.position.ndim == 1:
            self.state = ParticleState(
                position=self.position,
                velocity=self.velocity,
            )
        else:
            self.state = ParticleState(
                position=self.position[0],
                velocity=self.velocity[0],
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

fieldData = FieldMap(r"C:\Users\julia\OneDrive\Documents\Hackathon Gemelos Digitales\DraftHackathon\Hackathon_student\Electrode info\RK4 Sim\simion_efield_output.csv")

fieldData.summary()

print("Coordinate grid shape:", fieldData.coords.shape)
print("Field vector shape:", fieldData.E.shape)
print("Combined coord+field shape:", fieldData.coord_field.shape)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --- Experiment bounds in mm ---
x_min, x_max = 0.0, 484.0
y_min, y_max = 0.0, 153.0
z_min, z_max = 0.0, 484.0

print(f"Experiment size (mm): x={x_min:.1f}-{x_max:.1f}, y={y_min:.1f}-{y_max:.1f}, z={z_min:.1f}-{z_max:.1f}")

# --- Original single-particle beam ---
species = IonSpecies(mass=1.0, charge=1.0)
beam_single = Beam(
    species=species,
    position=np.array([80.0, 70.0, 50.0]),
    velocity=np.array([0.0, 0.0, 50.0]),
)
trajectory_single = Trajectory(beam_single)

integrator = RK4Integrator(fieldData)
integrator.integrate(trajectory_single, dt=0.1, num_steps=100)

positions_single = np.array([state.position for state in trajectory_single.states])

# --- Extra beam with 100 particles and a small random spread ---
np.random.seed(7)
base_position = np.array([80.0, 70.0, 50.0])
position_spread = np.random.normal(loc=0.0, scale=1.5, size=(100, 3))
position_spread[:, 1] *= 0.5
start_positions = base_position + position_spread

base_velocity = np.array([0.0, 0.0, 50.0])
velocity_spread = np.random.normal(loc=0.0, scale=0.8, size=(100, 3))
velocity_spread[:, 0] *= 0.3
velocity_spread[:, 1] *= 0.2
start_velocities = base_velocity + velocity_spread

beam_multi = Beam(
    species=species,
    position=start_positions,
    velocity=start_velocities,
)

positions_multi = []
for i in range(beam_multi.n_particles):
    particle_beam = Beam(
        species=species,
        position=start_positions[i],
        velocity=start_velocities[i],
    )
    particle_traj = Trajectory(particle_beam)
    integrator.integrate(particle_traj, dt=0.1, num_steps=100)
    positions_multi.append(np.array([state.position for state in particle_traj.states]))

positions_multi = np.array(positions_multi)

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

# plot the original single-particle trajectory through the full coordinate space
ax.plot(positions_single[:, 0], positions_single[:, 1], positions_single[:, 2], color="black", lw=1.5, label="single-particle path")
ax.scatter(positions_single[0, 0], positions_single[0, 1], positions_single[0, 2], color="lime", s=60, label="single start")
ax.scatter(positions_single[-1, 0], positions_single[-1, 1], positions_single[-1, 2], color="red", s=60, label="single end")

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
print(f"{'Step':>6} {'x':>12} {'y':>12} {'z':>12}")
for i, pos in enumerate(positions_single):
    print(f"{i:6d} {pos[0]:12.6e} {pos[1]:12.6e} {pos[2]:12.6e}")

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