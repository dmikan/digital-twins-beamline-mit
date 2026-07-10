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
        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] == 7:
            field_col_start = 4
            self.potential = data[:, 3]
        elif data.shape[1] == 6:
            field_col_start = 3
            self.potential = None
        else:
            raise ValueError(
                f"Expected 6 or 7 columns in {self.filename}, but found {data.shape[1]}."
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
        self.Ex = data[:, field_col_start].reshape(self.Nx, self.Ny, self.Nz)
        self.Ey = data[:, field_col_start + 1].reshape(self.Nx, self.Ny, self.Nz)
        self.Ez = data[:, field_col_start + 2].reshape(self.Nx, self.Ny, self.Nz)

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

        # Interpolate
        E = self.interpolator(positions)
        print(E)

        # Restore original shape
        return E.reshape(*original_shape, 3)
    
class IonSPecies:
    def __init__(self, mass, charge):
        self.mass = mass
        self.charge = charge

class ParticleState:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity

class Trajectory:
    def __init__(self, species, initial_state):
        self.species = species
        self.initial_state = initial_state
        self.states = [initial_state]  # List of ParticleState instances

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

        for _ in range(num_steps):
            k1 = self._derivative(state)
            k2 = self._derivative(ParticleState(state.position + 0.5 * dt * k1[0],
                                                state.velocity + 0.5 * dt * k1[1]))
            k3 = self._derivative(ParticleState(state.position + 0.5 * dt * k2[0],
                                                state.velocity + 0.5 * dt * k2[1]))
            k4 = self._derivative(ParticleState(state.position + dt * k3[0],
                                                state.velocity + dt * k3[1]))

            new_position = state.position + (dt / 6) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
            new_velocity = state.velocity + (dt / 6) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])

            state = ParticleState(new_position, new_velocity)
            trajectory.add_state(state)

    def _derivative(self, state):
        """
        Compute the derivative of the particle's state.

        Parameters
        ----------
        state : ParticleState
            The current state of the particle.

        Returns
        -------
        tuple
            A tuple containing the derivative of position and velocity.
        """
        E_field = self.field_map.field(state.position)
        acceleration = trajectory.species.charge * E_field / trajectory.species.mass
        return (state.velocity, acceleration)

fieldData = FieldMap(r"C:\Users\julia\OneDrive\Documents\Hackathon Gemelos Digitales\DraftHackathon\Hackathon_student\Electrode info\RK4 Sim\simion_efield_output.csv")

fieldData.summary()

print("Coordinate grid shape:", fieldData.coords.shape)
print("Field vector shape:", fieldData.E.shape)
print("Combined coord+field shape:", fieldData.coord_field.shape)

E_interp = fieldData.field(fieldData.coords) # Example position to test the field method

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