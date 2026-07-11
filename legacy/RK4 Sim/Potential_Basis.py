"""
BasisFieldMap
=============

Loads a set of per-electrode SIMION "basis" potential exports
(basis_electrode_1.csv, basis_electrode_2.csv, ...) ONCE, then lets you
recombine them cheaply (no file re-reading) any time the electrode
voltages change:

    V(x,y,z) = sum_i  voltage_i * V_i(x,y,z)

where V_i is electrode i's unit-voltage basis potential (loaded from
basis_electrode_i.csv). From the combined potential it computes the
electric field E = -grad(V) on the same grid, and exposes a fast
interpolator you can hand off to your RK4Integrator (same interface
as FieldMap.field() in RK4_sim_claude.py).

Typical usage
-------------
    bfm = BasisFieldMap.from_directory(
        "/path/to/csvs",
        pattern="basis_electrode_{i}.csv",
        n_electrodes=19,
    )

    voltages = np.zeros(19)
    voltages[2] = 150.0     # electrode 3 at 150 V
    voltages[7] = -50.0     # electrode 8 at -50 V

    bfm.set_voltages(voltages)          # recombine + recompute E field
    E = bfm.field([[100.0, 50.0, 200.0]])   # sample E at a point
"""

from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator


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
        self.coords = None

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
        directory = Path(directory)
        files = []
        for i in range(start_index, start_index + n_electrodes):
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

        # Coordinates grid shape: (Nx, Ny, Nz, 3)
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


def field(x, y, z, V):
    """Compute the electric field from a scalar potential on a regular grid.

    Parameters
    ----------
    x, y, z : 1D arrays
        Grid coordinates for each axis.
    V : ndarray, shape (Nx, Ny, Nz)
        Combined potential from all electrodes.

    Returns
    -------
    ndarray, shape (Nx, Ny, Nz, 3)
        Electric field vectors [Ex, Ey, Ez].
    """
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    dy = y[1] - y[0] if len(y) > 1 else 1.0
    dz = z[1] - z[0] if len(z) > 1 else 1.0

    dVdx, dVdy, dVdz = np.gradient(V, dx, dy, dz)
    return np.stack((-dVdx, -dVdy, -dVdz), axis=-1)


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


if __name__ == "__main__":
    # Minimal smoke test using whatever basis files are in the current directory.
    # Adjust n_electrodes / pattern to match what's actually available.
    import sys

    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    n_electrodes = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    bfm = BasisFieldMap.from_directory(directory, n_electrodes=n_electrodes)
    print("Field shape:", bfm.field(bfm.coords).shape)
    bfm.summary()

    voltages = np.zeros(n_electrodes)
    voltages[0] = 100.0
    bfm.set_voltages(voltages)
    print("\nAfter setting electrode 1 = 100V:")
    print("Field at grid center:", bfm.field([[bfm.x.mean(), bfm.y.mean(), bfm.z.mean()]]))

    # -------------------------------
    # Demo block: random voltage effects
    # Comment out the next line if you do not want this demo to run.
    # -------------------------------
    random_voltage_demo(bfm, n_samples=10, voltage_range=(-1000.0, 1000.0))