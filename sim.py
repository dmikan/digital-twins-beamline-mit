import numpy as np
import pandas as pd
import optimizer as op
import pathlib
from pathlib import Path
import optuna
import RK4_sim_basis as rk

# se llama a optuna una sola vez
SIMION_INSTALL_DIR = pathlib.Path(r"C:\Program Files\SIMION-8.1")

float_distribution = optuna.distributions.FloatDistribution(low=-1000.0, high=1000.0)

study = optuna.create_study(direction=op.DIRECTION,
           storage=f"sqlite:///{op.RESULTS_DB}",
           study_name=op.STUDY_NAME,
           load_if_exists=True)
sampler = study.sampler
N = 100

suggested_values = np.zeros((19, 100))
all_volts = {**op.FIXED, **op.OPTIMIZE}
for ii in range(N):
    for key, value in all_volts.items():
        if isinstance(value, float):
            suggested_values[key-1, ii] = value
        else:
            suggested_values[key-1, ii] = sampler.sample_independent(study, trial=None, param_name="x", param_distribution=float_distribution)


#llamar a lua para construir el array
cmd = f'simion --nogui lua ./extract_PA.lua'

#op.run_simion(cmd)

# se utilizan el array de voltajes obtenidos de optuna para correr rk4

n_electrodos = 19
field_map = rk.BasisFieldMap.from_directory(directory="./", n_electrodes=n_electrodos)

# --- Extra beam with N particles and a gaussian spread ---
    # -------------------------------
    # Beam parameters
    # -------------------------------
N = 500

u = 1.66053906660e-27      # kg
e = 1.602176634e-19        # C

mass = 28 * u
charge = 1 * e

species = rk.IonSpecies(
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


for ii in range(suggested_values.shape[1]):
    field_map.set_voltages(suggested_values[:,ii])
    integrator = rk.RK4Integrator(field_map)
    
    import time
    start_time = time.time()
    beam_multi = rk.Beam(
        species=species,
        position=start_positions,
        velocity=start_velocities,
    )
    multi_beam_traj = rk.Trajectory(beam_multi)
    integrator.integrate(multi_beam_traj, dt, num_steps)
    fin = time.time()-start_time
    print(fin)

    positions_multi = np.array([
        state.position
        for state in multi_beam_traj.states
    ])
    disp = positions_multi[1] - positions_multi[0]

# se utilizan los 10 mejores para correr en simion y obtener el out.txt



# se optimizan con optuna