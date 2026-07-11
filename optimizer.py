"""
optimizer.py
============

Unified optimization, configuration, cost planning, and starting points module.
Handles:
  - optimizer.py (configs, Optuna objective contract, basic SIMION wrappers)
  - orchestrator.py (loop logic, GP seeding, RK4 screening proxy, duplicate filtering)
  - starting_point.py (derived starting points loading and injection)
  - cost_planner.py ( screening fidelity and time-cost calculations)
"""

import json
import pathlib
import re
import subprocess
import time
import numpy as np
import optuna
from optuna import Trial

from physics import IonSpecies, BatchBasisFieldMap, make_batch_beam, BatchTrajectory, BatchRK4Integrator, cargar_fisica
from caracterizador import make_beam, BeamProgressScorer, caracterizar, objetivo_v2, desde_simion_ultimo_vuelo, cinematica_en_plano, puntaje_cinematico, distancias_al_detector

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SIM_FOLDER = ROOT
IOB_FILE = SIM_FOLDER / "SimpleSetUp.iob"
PA0_FILE = SIM_FOLDER / "electrode_.PA0"

SIMION_INSTALL_DIR = pathlib.Path(r"C:\Program Files\SIMION-8.1")
SIMION_EXE = SIMION_INSTALL_DIR / "simion.exe"

RESULTS_DB = ROOT / "studies" / "gemelo_v2.db"
RESULTS_CSV = ROOT / "studies" / "gemelo_v2_results.csv"
STUDY_NAME = "gemelo_v2"
REGISTRO = ROOT / "studies" / "registro_corridas.jsonl"
STARTING_POINT_FILE = ROOT / "derived_starting_point.json"

N_TRIALS = 50
OBJECTIVE = "J_v2.4 normalizado [0-1] sin pared (caracterizador.objetivo_v2, LAMBDA_HITS=2)"
DIRECTION = "minimize"
# v2.4 re-normaliza por (1+lambda): el peor J valido es 1.0 sin pared,
# ~1.05 con pared. BAD_SCORE debe superar ese tope (con 1.0 exacto, un
# vuelo FALLIDO de SIMION empataria con un config valido de 0 hits).
BAD_SCORE = 2.0

OPTIMIZE = {
    3:  (-1000.0, 1000.0),   # Einzel lens 1 (center)
    6:  (-1000.0, 1000.0),   # Einzel lens 2 (center)
    9:  (-1000.0, 1000.0),   # Quadrupole bender
    10: (-1000.0, 1000.0),   # Quadrupole bender
    11: (-1000.0, 1000.0),   # Quadrupole bender
    12: (-1000.0, 1000.0),   # Quadrupole bender
    15: (-1000.0, 1000.0),   # Voltage / deflection plate 1
    18: (-1000.0, 1000.0),   # Voltage / deflection plate 2
}

# Physical electrode numbers that are free (not fixed) -- used wherever a
# voltage vector must be sent to SIMION/physics.py, independent of how the
# SEARCH space parametrizes them (8 raw electrodes, or the reduced 6D
# bender reparametrization below).
OPTIMIZE_ELECTRODES = (3, 6, 9, 10, 11, 12, 15, 18)

# ----------------------------------------------------------------------
# Reduced 6D bender reparametrization (docs/INSTRUCCIONES_BENDER_6D.md)
# ----------------------------------------------------------------------
# PCA on 130 real hit-producing trials (gemelo_v2.db + gemelo_db_v2.db)
# showed 2 directions explain 98.5% of the variance in (V9,V10,V11,V12).
# The best PURELY ANALYTIC pair (geometry + one field measurement, zero
# trial history) is monopole (+1,+1,+1,+1, common bender offset) + quad
# (+1,-1,-1,+1, the diagonal-pair pattern already derived in
# tools/bender_field_analysis.py from which pair pushes the beam toward
# +z) -- together 81.6% of real variance, vs ~19% for any pair without
# the monopole term. Fixed linear map (no base term -- A already covers
# any offset):
#   V9  = A + B
#   V10 = A - B
#   V11 = A - B
#   V12 = A + B
# where A = common bender offset, B = bending-pattern strength.
BENDER_PLUS = (9, 12)
BENDER_MINUS = (10, 11)

REDUCED_OPTIMIZE = {
    3:  (-1000.0, 1000.0),   # Einzel lens 1 (center)
    6:  (-1000.0, 1000.0),   # Einzel lens 2 (center)
    "A": (-600.0, 600.0),    # bender common offset
    "B": (-800.0, 800.0),    # bender bending strength (diagonal pattern)
    "C": (-600.0, 600.0),    # bender horizontal dipole asymmetry strength
    15: (-1000.0, 1000.0),   # Voltage / deflection plate 1
    18: (-1000.0, 1000.0),   # Voltage / deflection plate 2
}
# NOTE (widened 2026-07-08): a first run with A,B in +-500 showed the best
# configs found kept drifting toward more negative B (-244 -> -382) without
# plateauing -- the +-500 bound was likely truncating the search right as
# it started to converge. Widened to A:+-600, B:+-800. Since A+B or A-B can
# now exceed the physical +-1000 electrode range, expand_reduced_to_full
# clips V9..V12 to +-1000 after expansion (see below).

def expand_bender(A, B, C=0.0):
    """A,B,C (offset, bending strength, horizontal asymmetry) -> physical V9..V12 dict."""
    return {9: A + B + C, 10: A - B, 11: A - B, 12: A + B - C}

def collapse_bender(v9, v10, v11, v12):
    """Physical V9..V12 -> best-fit (A, B, C) on the 3D bender subspace.

    Exact if the point already lies on the subspace; otherwise the L2
    projection (least-squares fit of A,B,C given the fixed map above).
    """
    A = (v9 + v10 + v11 + v12) / 4.0
    B = (v9 - v10 - v11 + v12) / 4.0
    C = (v9 - v12) / 2.0
    return A, B, C

def expand_reduced_to_full(reduced: dict) -> np.ndarray:
    """Reduced params (keys 3, 6, "A", "B", "C", 15, 18) -> full 19-electrode vector."""
    v = np.zeros(19)
    for k, val in FIXED.items():
        v[k - 1] = val
    v[3 - 1] = reduced[3]
    v[6 - 1] = reduced[6]
    A = reduced["A"]
    B = reduced["B"]
    C = reduced.get("C", 0.0)
    for e, val in expand_bender(A, B, C).items():
        v[e - 1] = float(np.clip(val, -1000.0, 1000.0))
    v[15 - 1] = reduced[15]
    v[18 - 1] = reduced[18]
    return v

FIXED = {
    1:  500.0,     # HV source
    19: -2000.0,   # Detector
    2:  0.0,       # pipe
    4:  0.0, 5: 0.0, 7: 0.0, 8: 0.0,    # Einzel outer rings (grounded)
    13: 0.0, 14: 0.0, 16: 0.0, 17: 0.0, # ground plates
}

DETECTOR_REGION = {"x": (70, 82), "y": (70, 83), "z": (403, 407)}

# Loop and Screening Tuning Parameters
DT = 1e-8
DETECTOR_BBOX = (69.0, 81.0, 68.5, 81.5, 403.0, 407.0)
WALL_EXCLUDE = (1, 2, 19)
WALL_HIT_MARGIN = 1.5
SCORE_WEIGHTS = dict(target_weight=1.0, wall_weight=1.0, lost_weight=0.3, target_scale=20.0)

NARROW_AFTER_HITS = 1
HIT_TERM_SCALE_MM = 20.0
RESCREEN_MIN = 30
RESCREEN_PARTICLES = 200
RESCREEN_SEED = 1234
RESCREEN_DT = 5e-9
RESCREEN_STEPS = 3000
RESCREEN_CHUNK = 8
WALL_PENALTY_MM = 50.0

KIN_PLANES = ((2, +1, 390.0),)
KIN_PENALTY_MM = 20.0
SPLAT_TOP_FRACTION = 0.10

# Starting Point Injection Settings
EXPLOIT_FRACTION = 0.4
PERTURBATION_STD = 150.0
BEST_FRACTION = 0.3
BEST_PERTURBATION_STD = 40.0

# Cost Planner fitting settings
SIMION_COST_PER_CANDIDATE_S = 5.77
RK4_FIXED_COST_PER_STEP_S = 1.80e-4
RK4_MARGINAL_COST_PER_PARTICLE_STEP_S = 2.80e-6
SCORING_OVERHEAD_MULTIPLIER = 1.14
MIN_VIABLE_STEPS = 1300
RECOMMENDED_SCREENING_STEPS = 1500
RECOMMENDED_SCREENING_PARTICLES = 50

# ----------------------------------------------------------------------
# SIMION Interfaces & Commands
# ----------------------------------------------------------------------
FLY_COMMAND = (
    f'"{SIMION_EXE}" --nogui fly --recording-output=out.txt --programs=0 '
    f'--retain-trajectories=0 --restore-potential=0 "{IOB_FILE}"'
)

def run_simion(command: str) -> str:
    result = subprocess.run(
        command, cwd=str(SIMION_INSTALL_DIR), shell=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
    )
    return result.stdout

def apply_voltages(chosen: dict) -> None:
    all_volts = {**FIXED, **chosen}
    settings = ",".join(f"{n}={v}" for n, v in sorted(all_volts.items()))
    run_simion(f'"{SIMION_EXE}" --nogui fastadj "{PA0_FILE}" {settings}')

def get_positions(simion_output: str) -> np.ndarray:
    pattern = r"xyz\(\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)mm"
    matches = re.findall(pattern, simion_output)
    return np.array(matches, dtype=float)

def _in_detector_region(positions: np.ndarray) -> np.ndarray:
    x_min, x_max = DETECTOR_REGION["x"]
    y_min, y_max = DETECTOR_REGION["y"]
    z_min, z_max = DETECTOR_REGION["z"]
    return (
        (positions[:, 0] > x_min) & (positions[:, 0] < x_max) &
        (positions[:, 1] > y_min) & (positions[:, 1] < y_max) &
        (positions[:, 2] > z_min) & (positions[:, 2] < z_max)
    )

def _detector_hits(positions: np.ndarray) -> np.ndarray:
    return positions[_in_detector_region(positions)]

def count_hits(positions: np.ndarray) -> int:
    hits = _detector_hits(positions).shape[0]
    print(f"  ions on detector: {hits}")
    return hits

def beam_spread(positions: np.ndarray) -> float:
    on_detector = _detector_hits(positions)
    if on_detector.shape[0] == 0:
        print("  no ions reached the detector")
        return BAD_SCORE
    spread = float(np.std(on_detector[:, 1]) + np.std(on_detector[:, 2]))
    print(f"  beam spread (y+z std): {spread:.3f}")
    return spread

def check_setup() -> None:
    missing = [p for p in (SIMION_INSTALL_DIR, IOB_FILE) if not p.exists()]
    if missing:
        print("Could not find these required SIMION files/folders:")
        for p in missing:
            print(f"   - {p}")
        raise SystemExit(1)
    if not PA0_FILE.exists():
        pa_define = ROOT / "electrode_.PA#"
        print(f"The field file '{PA0_FILE.name}' does not exist yet. Please refine the geometry first.")
        raise SystemExit(1)

# ----------------------------------------------------------------------
# Math utilities for screening
# ----------------------------------------------------------------------
def splat_distances(positions: np.ndarray) -> np.ndarray:
    return distancias_al_detector(positions)

def mean_splat_distance(positions: np.ndarray, top_fraction: float = 1.0) -> float:
    d = splat_distances(positions)
    if top_fraction < 1.0:
        n_keep = max(1, int(np.ceil(len(d) * top_fraction)))
        d = np.sort(d)[:n_keep]
    return float(d.mean())

# ----------------------------------------------------------------------
# Starting Point Loader & Injector
# ----------------------------------------------------------------------
def load_starting_point(n_electrodes=19):
    # Unbiased zero-start: all optimizable voltages set to 0.0 V,
    # keeping only the fixed hardware limits (HV source, grounded plates, detector)
    v = np.zeros(n_electrodes)
    for key, value in FIXED.items():
        v[key - 1] = float(value)
    return v

def inject_starting_point(voltages_batch, optimize, rng=None,
                           exploit_fraction=EXPLOIT_FRACTION,
                           perturbation_std=PERTURBATION_STD,
                           best_center=None, best_fraction=BEST_FRACTION,
                           best_std=BEST_PERTURBATION_STD,
                           bounds_dict=None):
    if bounds_dict is None:
        bounds_dict = optimize
    sp = load_starting_point(voltages_batch.shape[1])
    if sp is None and best_center is None:
        return 0, 0
    rng = rng or np.random.default_rng()
    m = voltages_batch.shape[0]

    n_best = 0
    if best_center is not None:
        n_best = max(1, min(m, int(round(m * best_fraction))))
        exploit_fraction = exploit_fraction / 2.0
    n_sp = 0
    if sp is not None:
        n_sp = max(1, min(m - n_best, int(round(m * exploit_fraction))))

    row = 0
    for _ in range(n_best):
        for e, (low, high) in bounds_dict.items():
            value = best_center[e - 1] + rng.normal(0.0, best_std)
            voltages_batch[row, e - 1] = np.clip(value, low, high)
        row += 1
    for j in range(n_sp):
        for e, (low, high) in bounds_dict.items():
            value = sp[e - 1]
            if j > 0:
                value += rng.normal(0.0, perturbation_std)
            voltages_batch[row, e - 1] = np.clip(value, low, high)
        row += 1
    return n_sp, n_best

def _full_to_reduced_row(v: np.ndarray) -> dict:
    """Full 19-electrode vector -> reduced dict {3,6,"A","B","C",15,18}."""
    A, B, C = collapse_bender(v[9 - 1], v[10 - 1], v[11 - 1], v[12 - 1])
    return {3: v[3 - 1], 6: v[6 - 1], "A": A, "B": B, "C": C, 15: v[15 - 1], 18: v[18 - 1]}

def inject_starting_point_reduced(voltages_batch, reduced_batch, rng=None,
                                   exploit_fraction=EXPLOIT_FRACTION,
                                   perturbation_std=PERTURBATION_STD,
                                   best_center=None, best_fraction=BEST_FRACTION,
                                   best_std=BEST_PERTURBATION_STD,
                                   bounds_dict=None):
    """Same as inject_starting_point, but perturbs in the reduced (A,B)
    space and keeps voltages_batch/reduced_batch in sync (both indexed by
    the same rows). sp/best_center are still full 19-electrode vectors
    (derived_starting_point.json and history are physical); they get
    collapsed to (A,B) here before perturbation."""
    if bounds_dict is None:
        bounds_dict = REDUCED_OPTIMIZE
    sp_full = load_starting_point(voltages_batch.shape[1])
    if sp_full is None and best_center is None:
        return 0, 0
    rng = rng or np.random.default_rng()
    m = voltages_batch.shape[0]

    n_best = 0
    if best_center is not None:
        n_best = max(1, min(m, int(round(m * best_fraction))))
        exploit_fraction = exploit_fraction / 2.0
    n_sp = 0
    if sp_full is not None:
        n_sp = max(1, min(m - n_best, int(round(m * exploit_fraction))))

    def write_row(row, reduced):
        voltages_batch[row] = expand_reduced_to_full(reduced)
        reduced_batch[row] = reduced

    row = 0
    if n_best:
        center_reduced = _full_to_reduced_row(best_center)
    for _ in range(n_best):
        reduced = {}
        for k, (low, high) in bounds_dict.items():
            value = center_reduced[k] + rng.normal(0.0, best_std)
            reduced[k] = float(np.clip(value, low, high))
        write_row(row, reduced)
        row += 1
    if n_sp:
        sp_reduced = _full_to_reduced_row(sp_full)
    for j in range(n_sp):
        reduced = {}
        for k, (low, high) in bounds_dict.items():
            value = sp_reduced[k]
            if j > 0:
                value += rng.normal(0.0, perturbation_std)
            reduced[k] = float(np.clip(value, low, high))
        write_row(row, reduced)
        row += 1
    return n_sp, n_best

# ----------------------------------------------------------------------
# Cost Planner
# ----------------------------------------------------------------------
def rk4_batch_cost(M, N=RECOMMENDED_SCREENING_PARTICLES, T=RECOMMENDED_SCREENING_STEPS,
                    a=RK4_FIXED_COST_PER_STEP_S, b=RK4_MARGINAL_COST_PER_PARTICLE_STEP_S,
                    include_scoring=True):
    integration = T * a + T * b * N * M
    return integration * SCORING_OVERHEAD_MULTIPLIER if include_scoring else integration

def plan_batch(simion_budget_k, rk4_time_budget_s=None, total_time_budget_s=None,
               rk4_particles=RECOMMENDED_SCREENING_PARTICLES,
               rk4_steps=RECOMMENDED_SCREENING_STEPS,
               simion_cost=SIMION_COST_PER_CANDIDATE_S,
               min_subset_fraction=0.01, max_subset_fraction=0.5):
    if rk4_steps < MIN_VIABLE_STEPS:
        raise ValueError(f"rk4_steps={rk4_steps} is below viability floor ({MIN_VIABLE_STEPS})")
    simion_time = simion_budget_k * simion_cost
    if rk4_time_budget_s is None:
        if total_time_budget_s is not None:
            rk4_time_budget_s = max(0.0, total_time_budget_s - simion_time)
        else:
            rk4_time_budget_s = simion_time

    fixed = rk4_steps * RK4_FIXED_COST_PER_STEP_S * SCORING_OVERHEAD_MULTIPLIER
    marginal_per_candidate = (rk4_steps * RK4_MARGINAL_COST_PER_PARTICLE_STEP_S
                               * rk4_particles * SCORING_OVERHEAD_MULTIPLIER)

    if rk4_time_budget_s <= fixed:
        raise ValueError(f"rk4_time_budget_s={rk4_time_budget_s:.2f}s too small for fixed overhead ({fixed:.2f}s)")

    M = int((rk4_time_budget_s - fixed) / marginal_per_candidate)
    M = max(M, simion_budget_k)

    if simion_budget_k / M > max_subset_fraction:
        M = int(np.ceil(simion_budget_k / max_subset_fraction))
    elif simion_budget_k / M < min_subset_fraction:
        M = int(np.floor(simion_budget_k / min_subset_fraction))

    predicted_screening_time = rk4_batch_cost(M, rk4_particles, rk4_steps)
    return {
        "rk4_particles": rk4_particles,
        "rk4_steps": rk4_steps,
        "M": M,
        "simion_budget_k": simion_budget_k,
        "subset_fraction": simion_budget_k / M,
        "predicted_screening_time_s": predicted_screening_time,
        "predicted_simion_time_s": simion_time,
        "predicted_total_time_s": predicted_screening_time + simion_time,
    }

# ----------------------------------------------------------------------
# Loop Generation & Optimization Methods
# ----------------------------------------------------------------------
def run_simion_candidate(voltages: np.ndarray) -> tuple:
    chosen = {k: float(voltages[k - 1]) for k in OPTIMIZE_ELECTRODES}
    try:
        apply_voltages(chosen)
        out = run_simion(FLY_COMMAND)
    except subprocess.CalledProcessError:
        return BAD_SCORE, 0, 1.0e6, {}

    positions = get_positions(out)
    if positions.shape[0] == 0:
        return BAD_SCORE, 0, 1.0e6, {}
    try:
        _, features = desde_simion_ultimo_vuelo()
    except Exception:
        features = caracterizar(positions)
    value, _ = objetivo_v2(features, con_pared=False)
    hits = int(features.get("hits") or count_hits(positions))
    return float(value), hits, mean_splat_distance(positions), features

def sample_voltage_batch(study, m, bounds_dict=None):
    if bounds_dict is None:
        bounds_dict = OPTIMIZE
    sampler = study.sampler
    all_volts = {**FIXED}
    for e, b in bounds_dict.items():
        all_volts[e] = b
    batch = np.zeros((m, 19))
    for i in range(m):
        for key, value in all_volts.items():
            if isinstance(value, float):
                batch[i, key - 1] = value
            else:
                low, high = value
                dist = optuna.distributions.FloatDistribution(low=low, high=high)
                batch[i, key - 1] = sampler.sample_independent(
                    study, trial=None, param_name=f"V{key}", param_distribution=dist)
    return batch

def sample_voltage_batch_gp_seeded(study, m, n_seeds, perturbation_std, rng_seed=None, bounds_dict=None):
    if bounds_dict is None:
        bounds_dict = OPTIMIZE
    rng = np.random.default_rng(rng_seed)
    n_seeds = max(1, min(n_seeds, m))
    optimize_keys = sorted(OPTIMIZE.keys())
    voltages_batch = np.zeros((m, 19))
    for k, v in FIXED.items():
        voltages_batch[:, k - 1] = v

    seed_trial_for_row = {}
    for i in range(n_seeds):
        trial = study.ask()
        for e in optimize_keys:
            # ALWAYS suggest using the original OPTIMIZE static bounds to avoid GPSampler fallback warnings
            low, high = OPTIMIZE[e]
            voltages_batch[i, e - 1] = trial.suggest_float(f"V{e}", low, high)
        seed_trial_for_row[i] = trial

    remaining = m - n_seeds
    for j in range(remaining):
        row = n_seeds + j
        base = j % n_seeds
        voltages_batch[row] = voltages_batch[base].copy()
        for e in optimize_keys:
            low, high = bounds_dict[e]
            noisy = voltages_batch[row, e - 1] + rng.normal(0, perturbation_std)
            voltages_batch[row, e - 1] = np.clip(noisy, low, high)

    return voltages_batch, seed_trial_for_row

def _param_name(key):
    return key if isinstance(key, str) else f"V{key}"

def sample_voltage_batch_reduced(study, m, bounds_dict=None):
    """Reduced 6D equivalent of sample_voltage_batch: samples {3,6,"A",
    "B",15,18} from the sampler and expands to a full 19-electrode row.
    Returns (voltages_batch (m,19), reduced_batch (list of dict))."""
    if bounds_dict is None:
        bounds_dict = REDUCED_OPTIMIZE
    sampler = study.sampler
    dists = {k: optuna.distributions.FloatDistribution(low=lo, high=hi)
             for k, (lo, hi) in bounds_dict.items()}
    voltages_batch = np.zeros((m, 19))
    reduced_batch = []
    for i in range(m):
        reduced = {}
        for key in bounds_dict:
            reduced[key] = sampler.sample_independent(
                study, trial=None, param_name=_param_name(key), param_distribution=dists[key])
        voltages_batch[i] = expand_reduced_to_full(reduced)
        reduced_batch.append(reduced)
    return voltages_batch, reduced_batch

def sample_voltage_batch_gp_seeded_reduced(study, m, n_seeds, perturbation_std, rng_seed=None, bounds_dict=None):
    """Reduced 6D equivalent of sample_voltage_batch_gp_seeded. The
    sampler (TPE/GP) asks/suggests directly in (A,B) space -- this is the
    whole point of the reparametrization, fewer dims for the sampler's
    model. Returns (voltages_batch, reduced_batch, seed_trial_for_row)."""
    if bounds_dict is None:
        bounds_dict = REDUCED_OPTIMIZE
    rng = np.random.default_rng(rng_seed)
    n_seeds = max(1, min(n_seeds, m))
    voltages_batch = np.zeros((m, 19))
    reduced_batch = [None] * m

    seed_trial_for_row = {}
    for i in range(n_seeds):
        trial = study.ask()
        # ALWAYS suggest using the original REDUCED_OPTIMIZE static bounds to avoid GPSampler fallback warnings
        reduced = {key: trial.suggest_float(_param_name(key), lo, hi)
                   for key, (lo, hi) in REDUCED_OPTIMIZE.items()}
        voltages_batch[i] = expand_reduced_to_full(reduced)
        reduced_batch[i] = reduced
        seed_trial_for_row[i] = trial

    remaining = m - n_seeds
    for j in range(remaining):
        row = n_seeds + j
        base = j % n_seeds
        reduced = dict(reduced_batch[base])
        for key, (low, high) in bounds_dict.items():
            noisy = reduced[key] + rng.normal(0, perturbation_std)
            reduced[key] = float(np.clip(noisy, low, high))
        voltages_batch[row] = expand_reduced_to_full(reduced)
        reduced_batch[row] = reduced

    return voltages_batch, reduced_batch, seed_trial_for_row

def rk4_score_chunk(bfm, wall_index, voltages_chunk, start_positions, start_velocities,
                     species, dt, num_steps, kinematica=True, detalle=False, iteration=1):
    # Dynamic weights scheduling based on iteration stage
    if iteration in (1, 2): # Exploration (first two rounds)
        local_weights = dict(target_weight=0.5, wall_weight=0.2, lost_weight=1.0, target_scale=20.0)
        local_kin_penalty = 0.0
        kinematica = False
    elif iteration == 3: # Transition / Coarse Tuning
        local_weights = dict(target_weight=1.0, wall_weight=1.0, lost_weight=0.3, target_scale=20.0)
        local_kin_penalty = 10.0
    else: # iteration >= 4: Refinement (last three rounds)
        local_weights = dict(target_weight=1.0, wall_weight=2.0, lost_weight=0.1, target_scale=20.0)
        local_kin_penalty = 30.0

    n_configs = voltages_chunk.shape[0]
    n = start_positions.shape[0]

    bfm.set_voltages_batch(voltages_chunk)
    beam, config_indices = make_batch_beam(species, start_positions, start_velocities, n_configs)
    trajectory = BatchTrajectory(beam, config_indices)
    BatchRK4Integrator(bfm, config_indices).integrate(trajectory, dt=dt, num_steps=num_steps)

    scorer = BeamProgressScorer(
        bfm=bfm, Trajectory=trajectory, dt=dt, num_steps=num_steps,
        detector_bbox=DETECTOR_BBOX, wall_index=wall_index,
        wall_hit_margin=WALL_HIT_MARGIN,
        wall_check_midpoints=False, wall_check_stride=3,
    )
    result = scorer.combined_score(voltages_chunk, **local_weights)

    tdist = result["target_distance"].reshape(n_configs, n)
    reached_fraction = result["reached_target"].reshape(n_configs, n).mean(axis=1)
    wall_fraction = result["hit_wall"].reshape(n_configs, n).mean(axis=1)
    n_keep = max(1, int(np.ceil(n * SPLAT_TOP_FRACTION)))
    top_mean = np.sort(tdist, axis=1)[:, :n_keep].mean(axis=1)
    base = (top_mean
            + HIT_TERM_SCALE_MM * (1.0 - reached_fraction)
            + WALL_PENALTY_MM * wall_fraction)

    kin = np.ones(n_configs)
    if kinematica:
        posiciones = result["positions"].reshape(-1, n_configs, n, 3)
        stop_r = result["stop_idx"].reshape(n_configs, n)
        pared_r = result["hit_wall_step"].reshape(n_configs, n)
        estados = trajectory.states
        for c in range(n_configs):
            sl = slice(c * n, (c + 1) * n)
            vel_c = np.asarray([s.velocity[sl] for s in estados], dtype=np.float32)
            puntajes = [
                puntaje_cinematico(cinematica_en_plano(
                    posiciones[:, c], vel_c, stop_idx=stop_r[c],
                    paso_pared=pared_r[c], z_plano=plano, eje=eje,
                    direccion=direccion))
                for eje, direccion, plano in KIN_PLANES
            ]
            kin[c] = float(np.mean(puntajes))
        total = base + local_kin_penalty * kin
    else:
        total = base

    if detalle:
        return total, base, kin
    return total

def rk4_score_all(bfm, wall_index, voltages_batch, start_positions, start_velocities,
                   species, dt, num_steps, chunk_size, iteration=1):
    M = voltages_batch.shape[0]
    scores = np.empty(M)
    n_chunks = int(np.ceil(M / chunk_size))
    for c in range(n_chunks):
        lo, hi = c * chunk_size, min((c + 1) * chunk_size, M)
        scores[lo:hi] = rk4_score_chunk(bfm, wall_index, voltages_batch[lo:hi],
                                         start_positions, start_velocities, species, dt, num_steps,
                                         iteration=iteration)
    return scores

def build_study(sampler=None, name=STUDY_NAME, db_path=RESULTS_DB):
    return optuna.create_study(
        direction=DIRECTION,
        storage=f"sqlite:///{db_path}",
        study_name=name,
        load_if_exists=True,
        sampler=sampler,
    )

def select_diverse_k(candidates_idx, voltages_batch, k_needed, max_candidates_to_consider=30):
    """Greedy Max-Min diversity selection on candidates.
    
    Given a list of candidate indices (sorted by RK4 score), selects k_needed
    candidates that maximize the minimum Euclidean distance in parameter space
    to already selected candidates, considering up to max_candidates_to_consider.
    """
    if len(candidates_idx) <= k_needed:
        return list(candidates_idx)
    
    selected = [int(candidates_idx[0])]
    remaining = [int(i) for i in candidates_idx[1:max_candidates_to_consider]]
    
    while len(selected) < k_needed and remaining:
        best_candidate = None
        max_min_dist = -1.0
        for c in remaining:
            # compute min distance from c to all currently selected
            # normalized by 1000.0 V to keep distance scale consistent
            dist = min(
                np.linalg.norm(voltages_batch[c] - voltages_batch[s]) / 1000.0
                for s in selected
            )
            if dist > max_min_dist:
                max_min_dist = dist
                best_candidate = c
        if best_candidate is not None:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        else:
            break
            
    # If we already have enough, return selected
    if len(selected) >= k_needed:
        return selected[:k_needed]
            
    # If we couldn't select enough, pad with remaining candidates in original rank order
    for c in candidates_idx:
        c_int = int(c)
        if c_int not in selected:
            selected.append(c_int)
            if len(selected) == k_needed:
                break
    return selected

def orchestrate(total_simion_budget, simion_per_iteration, target_score=None,
                 sampler=None, n_gp_seeds=5, perturbation_std=150.0,
                 verbose=True, studyname=STUDY_NAME, db_path=RESULTS_DB,
                 reduced_search=False):
    """reduced_search=True switches the sampler-facing search space from the
    raw 8 electrodes {V3,V6,V9,V10,V11,V12,V15,V18} to the 6D bender
    reparametrization {V3,V6,A,B,V15,V18} (docs/INSTRUCCIONES_BENDER_6D.md).
    physics.py/SIMION are unaffected either way -- they always see the
    expanded full 19-electrode vector. Use a dedicated studyname/db_path
    when reduced_search=True: it is a different Optuna search space and
    must not mix with existing 8D studies."""
    global STUDY_NAME, RESULTS_DB
    STUDY_NAME = studyname
    RESULTS_DB = db_path
    check_setup()

    global WALL_HIT_MARGIN
    fis = cargar_fisica(ROOT, n_electrodes=19)
    bfm, wall_index, WALL_HIT_MARGIN = fis.bfm, fis.wall, fis.margen
    species = IonSpecies(mass=28 * 1.66053906660e-27, charge=1.602176634e-19)

    param_keys = list(REDUCED_OPTIMIZE) if reduced_search else list(OPTIMIZE)
    param_names = [_param_name(k) for k in param_keys]

    study = build_study(sampler=sampler, name=STUDY_NAME, db_path=RESULTS_DB)
    if sampler is None:
        sampler = study.sampler
    gp_seeded = isinstance(sampler, optuna.samplers.BaseSampler)

    simion_spent = 0
    iteration = 0
    while simion_spent < total_simion_budget:
        iteration += 1
        k = min(simion_per_iteration, total_simion_budget - simion_spent)
        plan = plan_batch(simion_budget_k=k)
        M, N, T = plan["M"], plan["rk4_particles"], plan["rk4_steps"]
        chunk_size = min(M, fis.chunk_screening)

        print(f"\n=== Iteration {iteration} (SIMION spent: {simion_spent}/{total_simion_budget}) ===")
        print(f"  Plan: screen M={M} candidates, promote top K={k} to SIMION")

        start_positions, start_velocities = make_beam(
            N=N, species=species, start_point=[395.0, 75.0, 77.0],
            mean_energy_eV=15.0, std_energy_eV=0.42466, half_angle_deg=15.0, seed=42,
        )

        # 1. Annealing and stage progress calculation
        max_iterations = int(np.ceil(total_simion_budget / simion_per_iteration))
        progress = (iteration - 1) / float(max_iterations - 1) if max_iterations > 1 else 0.0
        progress = min(1.0, max(0.0, progress))

        # Geometric halving schedule: 500 -> 250 -> 125 -> 62.5 -> 31.25 -> 15.6 -> floor of 15V
        decay_factor = 0.5 ** (iteration - 1)
        current_pert_std = max(15.0, 500.0 * decay_factor)
        current_best_std = max(15.0, 150.0 * decay_factor)

        # 2. Find best candidate so far and construct dynamic bounds
        best_center = None
        best_hits = 0
        best_hits_trial = None
        for t in study.get_trials(deepcopy=False):
            h = t.user_attrs.get("simion_hits")
            if h is not None and h > best_hits and all(pn in t.params for pn in param_names):
                best_hits, best_hits_trial = h, t

        bounds_dict = None
        if best_hits > NARROW_AFTER_HITS:
            best_center = np.zeros(19)
            for e_num, v_fixed in FIXED.items():
                best_center[e_num - 1] = v_fixed
            
            # Shrink factor: search space width drops from 100% (at iter 1) to 40% (at iter 6)
            shrink_factor = 1.0 - 0.6 * progress
            bounds_dict = {}
            if reduced_search:
                reduced_best = {key: best_hits_trial.params[_param_name(key)] for key in REDUCED_OPTIMIZE}
                best_center = expand_reduced_to_full(reduced_best)
                for key, (low, high) in REDUCED_OPTIMIZE.items():
                    width = high - low
                    half_width = (width / 2.0) * shrink_factor
                    new_low = max(low, reduced_best[key] - half_width)
                    new_high = min(high, reduced_best[key] + half_width)
                    bounds_dict[key] = (float(new_low), float(new_high))
            else:
                for e_num in OPTIMIZE:
                    best_val = best_hits_trial.params[f"V{e_num}"]
                    best_center[e_num - 1] = best_val
                    low, high = OPTIMIZE[e_num]
                    width = high - low
                    half_width = (width / 2.0) * shrink_factor
                    new_low = max(low, best_val - half_width)
                    new_high = min(high, best_val + half_width)
                    bounds_dict[e_num] = (float(new_low), float(new_high))
            
            print(f"  Dynamic bounds active around best config ({best_hits} hits) with shrink_factor={shrink_factor:.2f}")

        # 3. Candidate Sampling
        n_real_trials = len(study.get_trials(deepcopy=False))
        seed_trial_for_row = {}
        reduced_batch = None
        if gp_seeded:
            t0 = time.time()
            if reduced_search:
                voltages_batch, reduced_batch, seed_trial_for_row = sample_voltage_batch_gp_seeded_reduced(
                    study, M, n_gp_seeds, current_pert_std, rng_seed=iteration, bounds_dict=bounds_dict)
            else:
                voltages_batch, seed_trial_for_row = sample_voltage_batch_gp_seeded(
                    study, M, n_gp_seeds, current_pert_std, rng_seed=iteration, bounds_dict=bounds_dict)
            sampler_name = type(sampler).__name__
            print(f"  {sampler_name} sampling: {len(seed_trial_for_row)} seeds in {time.time()-t0:.1f}s")
        else:
            if reduced_search:
                voltages_batch, reduced_batch = sample_voltage_batch_reduced(study, M, bounds_dict=bounds_dict)
            else:
                voltages_batch = sample_voltage_batch(study, M, bounds_dict=bounds_dict)

        # 4. Start Point and Perturbations Injection
        inject_rng = np.random.default_rng(1000 * n_real_trials + iteration)
        n_gp_rows = len(seed_trial_for_row)
        if reduced_search:
            target_v = voltages_batch[n_gp_rows:] if gp_seeded else voltages_batch
            target_r = reduced_batch[n_gp_rows:] if gp_seeded else reduced_batch
            n_sp, n_best = inject_starting_point_reduced(
                target_v, target_r, rng=inject_rng, perturbation_std=current_pert_std,
                best_center=best_center, best_std=current_best_std, bounds_dict=bounds_dict)
            if gp_seeded:
                reduced_batch[n_gp_rows:] = target_r
        else:
            target = voltages_batch[n_gp_rows:] if gp_seeded else voltages_batch
            n_sp, n_best = inject_starting_point(
                target, OPTIMIZE, rng=inject_rng, perturbation_std=current_pert_std,
                best_center=best_center, best_std=current_best_std, bounds_dict=bounds_dict)
        if n_best:
            print(f"  Injected {n_best} perturbations around best ({best_hits} hits) + {n_sp} around SP")
            print(f"  Perturbation step size annealed: normal_std={current_pert_std:.1f}V, best_std={current_best_std:.1f}V")
        elif n_sp:
            print(f"  Injected starting point + {n_sp - 1} perturbations")

        t0 = time.time()
        scores = rk4_score_all(bfm, wall_index, voltages_batch, start_positions, start_velocities,
                                species, DT, T, chunk_size, iteration=iteration)
        print(f"  RK4 screen: {M} candidates in {time.time()-t0:.1f}s")

        n_rescreen = int(min(M, max(3 * k, RESCREEN_MIN)))
        shortlist = np.argsort(scores)[:n_rescreen]
        rescreen_positions, rescreen_velocities = make_beam(
            N=RESCREEN_PARTICLES, species=species, start_point=[395.0, 75.0, 77.0],
            mean_energy_eV=15.0, std_energy_eV=0.42466, half_angle_deg=15.0, seed=RESCREEN_SEED)
        t0 = time.time()
        scores_b = rk4_score_all(bfm, wall_index, voltages_batch[shortlist],
                                  rescreen_positions, rescreen_velocities, species,
                                  RESCREEN_DT, RESCREEN_STEPS, RESCREEN_CHUNK, iteration=iteration)
        print(f"  RK4 rescreen: top {n_rescreen} in {time.time()-t0:.1f}s")
        score_by_idx = {int(i): float(s) for i, s in zip(shortlist, scores_b)}
        promotion_order = shortlist[np.argsort(scores_b)]

        seen = {
            tuple(round(t.params[pn], 1) for pn in param_names)
            for t in study.get_trials(deepcopy=False)
            if all(pn in t.params for pn in param_names)
        }
        # 1. Get the top-k from RK4 screening (excluding duplicates and promoting diversity)
        filtered_promotion_order = []
        n_dupes_skipped = 0
        for idx in promotion_order:
            if reduced_search:
                key = tuple(round(reduced_batch[idx][k_key], 1) for k_key in REDUCED_OPTIMIZE)
            else:
                key = tuple(round(float(voltages_batch[idx][e - 1]), 1) for e in OPTIMIZE)
            if key in seen:
                n_dupes_skipped += 1
                continue
            seen.add(key)
            filtered_promotion_order.append(idx)
        
        # Select the most diverse k indices from the filtered list
        top_k_idx = select_diverse_k(filtered_promotion_order, voltages_batch, k, max_candidates_to_consider=30)

        # 2. Merge only with the SINGLE BEST GP seed based on RK4 score (if not already in top_k_idx)
        gp_seed_indices = list(seed_trial_for_row.keys())
        best_gp_seed_idx = min(gp_seed_indices, key=lambda idx: float(scores[idx]))
        
        promoted_indices = list(top_k_idx)
        if best_gp_seed_idx not in promoted_indices:
            promoted_indices.append(int(best_gp_seed_idx))
        promoted_indices = np.array(promoted_indices, dtype=int)

        if n_dupes_skipped:
            print(f"  Skipped {n_dupes_skipped} duplicates during promotion")
        if len(promoted_indices) == 0:
            break

        n_seeds_promoted = sum(1 for idx in promoted_indices if idx in seed_trial_for_row)
        sampler_name = type(sampler).__name__
        print(f"  Promoting top {len(promoted_indices)} to real SIMION ({n_seeds_promoted} {sampler_name} seeds)...")

        dist_opt = optuna.distributions.FloatDistribution(low=-1000.0, high=1000.0)
        for rank, idx in enumerate(promoted_indices, 1):
            voltages = voltages_batch[idx]
            t0 = time.time()
            real_value, simion_hits, mean_all, feats = run_simion_candidate(voltages)
            simion_spent += 1
            elapsed = time.time() - t0
            user_attrs = {
                "rk4_score": score_by_idx.get(idx, float(scores[idx])),
                "rk4_stage_a_score": float(scores[idx]),
                "iteration": iteration,
                "rk4_rank_in_iteration": rank,
                "simion_elapsed_s": elapsed,
                "gp_seed": bool(idx in seed_trial_for_row),
                "simion_hits": simion_hits,
                "mean_splat_all_mm": mean_all,
            }
            for fk in ("dist_punta_mm", "n_plane", "offset_x_mm", "offset_y_mm",
                       "sigma_x_mm", "sigma_y_mm", "halo_fraction", "kurtosis_x",
                       "kurtosis_y", "div_x_mrad", "div_y_mrad", "twiss_alpha_x",
                       "twiss_alpha_y", "emittance_x", "emittance_y",
                       "resid_transporte_x_mm", "resid_transporte_y_mm"):
                if fk in feats:
                    user_attrs[f"f_{fk}"] = float(feats[fk])

            if idx in seed_trial_for_row:
                trial = seed_trial_for_row[idx]
                for name, value in user_attrs.items():
                    trial.set_user_attr(name, value)
                study.tell(trial, real_value)
            else:
                if reduced_search:
                    reduced = reduced_batch[idx]
                    params = {_param_name(k): float(reduced[k]) for k in REDUCED_OPTIMIZE}
                    distributions = {_param_name(k): optuna.distributions.FloatDistribution(low=lo, high=hi)
                                     for k, (lo, hi) in REDUCED_OPTIMIZE.items()}
                else:
                    params = {f"V{e}": float(voltages[e - 1]) for e in OPTIMIZE}
                    distributions = {f"V{e}": dist_opt for e in OPTIMIZE}
                trial = optuna.trial.create_trial(
                    params=params, distributions=distributions, value=real_value,
                    user_attrs=user_attrs,
                )
                study.add_trial(trial)
            print(f"    [{rank}/{len(promoted_indices)}] J_real={real_value:.3f}, hits={simion_hits} ({elapsed:.1f}s)")

        # Prune non-promoted GP seeds so Optuna learns from the RK4 filtering without running SIMION
        for idx, trial in seed_trial_for_row.items():
            if idx not in promoted_indices:
                trial.set_user_attr("rk4_screened_out", True)
                trial.set_user_attr("rk4_score", float(scores[idx]))
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)

    return study
