"""
orchestrator.py
=================

Closes the loop that sim_batch.py stopped short of:

    sampler suggests M candidates
        -> RK4 batch-screens & ranks them (cheap)
        -> top-K run through REAL SIMION (expensive, ground truth)
        -> K real results fed back into the study
        -> repeat, now informed by real SIMION history
    ...until a SIMION-run budget is exhausted or a target score is reached.

Does NOT modify any of the files it depends on -- everything below is
either imported unchanged or newly written here.

REQUIRED FILES (all of these must sit alongside this script)
----------------------------------------------------------------
Code (imported, unmodified):
    RK4_sim_basis.py         IonSpecies
    RK4_sim_basis_batch.py   BatchBasisFieldMap, BatchRK4Integrator, BatchTrajectory, make_batch_beam
    beam_progress_score.py   make_beam, BeamProgressScorer
    electrode_geometry.py    build_wall_index
    cost_planner.py          plan_batch  (RK4-screening fidelity / M-vs-K time-cost model)
    optimizer.py             FIXED, OPTIMIZE, DIRECTION, BAD_SCORE, OBJECTIVE,
                              get_positions, count_hits, beam_spread, check_setup,
                              SIM_FOLDER, IOB_FILE, PA0_FILE, SIMION_INSTALL_DIR

Data (must be present in this same directory):
    basis_electrode_1.csv ... basis_electrode_19.csv   (RK4 field data)
    electrode_1.stl ... electrode_19.stl               (collision geometry)
    SimpleSetUp.iob                                    (SIMION "fly" simulation)
    electrode_.PA0                                     (SIMION potential array --
                                                         REWRITTEN by fastadj on every
                                                         real SIMION candidate, by design)

External:
    SIMION 8.1 installed (SIMION_INSTALL_DIR in optimizer.py, default
    C:\\Program Files\\SIMION-8.1). Uses the fully-qualified path to
    simion.exe, not the bare "simion" command -- optimizer.py's own
    run_simion() has a bug where that bare command isn't resolvable via
    subprocess(shell=True) in this environment; fixed here the same way
    optimizer_gp.py fixed it.

SAMPLER CHOICE: TPE by default; GPSampler is properly wired in, not just
swappable
--------------------------------------------------------------------------
Checked directly (see conversation): GPSampler.sample_independent() just
delegates to a plain RandomSampler fallback -- it does NOT consult the GP
model at all for the kind of cheap, per-parameter sampling that would
naively generate the M RK4-screening candidates every iteration. Only
real GP-informed suggestions come from ask()/sample_relative(), which
costs ~100-300ms+ per call and GROWS with trial count (see
compare_samplers.py) -- far too expensive to call M times (M is typically
in the hundreds; see cost_planner.py) every iteration.

So orchestrate() branches on sampler type (isinstance(sampler,
optuna.samplers.GPSampler)):
  - TPE (default, sampler=None): sample_voltage_batch() -- cheap
    sample_independent() calls for all M candidates. TPE's independent
    sampling DOES use trial history and stays flat-cost regardless of
    trial count, so this is fine as-is.
  - GPSampler: sample_voltage_batch_gp_seeded() -- pays the GP's real
    joint model (sample_relative(), via n_gp_seeds real study.ask() calls)
    only a HANDFUL of times per iteration, one sample_relative() call
    covering all 8 electrodes at once (verified: first suggest_float on a
    fresh trial triggers it and takes ~60ms with modest history; the
    remaining 7 suggest_float calls on that trial are ~0.2ms, cached).
    The rest of the M-candidate RK4-screening batch is cheap Gaussian
    perturbation (plain numpy) around those seeds. This is what actually
    lets the GP's posterior/uncertainty steer WHERE the batch
    concentrates -- much closer to "GP generates a candidate, RK4 explores
    more of the space [around it]" than uniform-random screening would be.
    Promoted seeds are told() directly (they're real asked trials); seeds
    that get asked but never promoted are explicitly pruned so they don't
    sit as dangling RUNNING trials in the persisted study.

Run with:
    python orchestrator.py
"""

import pathlib
import subprocess
import time

import numpy as np
import optuna

import optimizer as op
from RK4_sim_basis import IonSpecies
from RK4_sim_basis_batch import (
    BatchBasisFieldMap, make_batch_beam, BatchTrajectory, BatchRK4Integrator,
)
from beam_progress_score import make_beam, BeamProgressScorer
from electrode_geometry import build_wall_index
from cost_planner import plan_batch
from starting_point import inject_starting_point
from caracterizador import caracterizar, objetivo_v2, desde_simion_ultimo_vuelo

HERE = pathlib.Path(__file__).resolve().parent
N_ELECTRODES = 19
# 1e-8, not the historical 5e-8: with the corrected acceleration units
# (RK4_sim_basis_batch.E_VMM_TO_ACCEL_MM) the beam leaves the +500V source
# at ~470 eV, i.e. ~2.9mm per 5e-8 step -- far too coarse through the 44mm
# bender gap. 1e-8 gives ~0.6mm/step; the planner's 2500 steps then spans
# ~2.5x the full transit.
DT = 1e-8

DETECTOR_BBOX = (69.0, 81.0, 68.5, 81.5, 403.0, 407.0)
WALL_EXCLUDE = (1, 2, 19)
# target_scale was 30.0 -- see sim_batch.py's SCORE_WEIGHTS comment / this
# session's diagnosis of why combined_score has been ~flat across every
# batch so far (the reward term for detector proximity was contributing
# ~0 signal at the actual distances candidates land -- 200-350mm). 150mm
# fixes that.
SCORE_WEIGHTS = dict(target_weight=1.0, wall_weight=1.0, lost_weight=0.3, target_scale=150.0)

# Keep the search BROAD until a SIMION trial actually transmits a
# meaningful fraction of the beam. The tight exploit-around-best block
# used to switch on at a single ion on the detector, which is
# noise-level (identical voltages measured 2 vs 5 hits on repeat flies)
# -- and that prematurely narrowed the whole RK4-screening batch around
# a barely-working config. Until some trial puts more than this many
# ions (of 500) on the detector, the batch stays exploration-heavy
# (sampler + wide starting-point perturbations). Hits are read from each
# trial's "simion_hits" user attribute, NOT from trial.value -- the
# objective value is now a distance (see below).
NARROW_AFTER_HITS = 100

# SIMION OBJECTIVE: dense mean-splat-distance, not the sparse hit count.
# optimizer.py's OBJECTIVE="hits" gives an integer 0..500 that is 0 for
# almost every candidate and 0-10 even in the good basin -- between two
# zero-hit configs the sampler learns nothing, even if one splats 5mm
# from the detector and the other 300mm away. Instead each ion's splat
# position is scored by its Euclidean distance to the DETECTOR_REGION
# box (0 if it landed inside = a hit), averaged over all recorded ions:
# a smooth mm-scale value where EVERY ion contributes signal. Lower is
# better -> the study direction is "minimize" (overrides op.DIRECTION).
# Hit counts are still computed and stored per-trial in user_attrs
# ("simion_hits") for the NARROW_AFTER_HITS gate and reporting.
# NOTE: not comparable with the old maximize-hits studies -- use a NEW
# study/DB for runs with this objective.
# OBJETIVO VIVO desde 2026-07-05: J_v2 normalizado del caracterizador
# unico (ver caracterizador.objetivo_v2 y PESOS_V2). J en [0,1], 0 =
# perfecto; hits por encima de todo (0.45), luego acercamiento (0.15,
# unico gradiente lejos de la cuenca) y la escalera de forma
# offset>halo>kurtosis>colimacion>twiss. La formula vieja (best-10% mm)
# sigue disponible abajo (mean_splat_distance) para diagnosticos y
# porque el termino de acercamiento la reutiliza.
OBJECTIVE = "J_v2 normalizado [0-1] (caracterizador.objetivo_v2)"
DIRECTION = "minimize"
BAD_SCORE = 1.0  # el peor J posible: SIMION fallo o no registro iones

# Saturation fix, measured on the first best-10% smoke run: the hitting
# config's 50 closest ions all splat ON the detector surface, so its
# best-10% distance is exactly 0.0 with only 5 counted hits -- the
# distance term runs out of gradient precisely when transmission becomes
# the thing to optimize. A small hit-fraction term hands pressure over
# smoothly: far configs are ranked by distance (this adds a ~constant
# +20mm), saturated configs by how much of the beam actually lands
# inside the hit window (5/500 hits -> +19.8, 100 -> +16.0, all -> +0).
HIT_TERM_SCALE_MM = 20.0

# Two-stage screening. Stage A ranks all M candidates cheaply (the
# cost-planner fidelity: 50 particles, seed 42). Hit fractions at 50
# particles quantize in 2% steps -- too coarse to separate near-misses --
# so stage B re-flies only the stage-A shortlist at 4x the particles, a
# DIFFERENT beam seed (so promotion never overfits one particular draw of
# 50), halved dt, and doubled steps, and promotion ranks on THOSE scores.
RESCREEN_MIN = 30          # shortlist size: max(3*K, this)
RESCREEN_PARTICLES = 200
RESCREEN_SEED = 1234       # independent of stage A's seed 42
RESCREEN_DT = 5e-9
RESCREEN_STEPS = 3000      # same 3e-5s of flight as stage A's 1500 @ 1e-8
RESCREEN_CHUNK = 15        # memory: ~430MB peak per chunk at this fidelity

# Wall-proximity penalty in the RK4 screening score, in mm-equivalents.
# Calibrated OFFLINE against 288 archived voltage sets with known SIMION
# outcomes (validate_rk4_filter.py, 2026-07-03): terminating particles at
# wall contact LOST at every margin tried (7-9/61 real hitters ranked in
# the top-30, vs 16-17 without -- kill locations inherit the ~11-15mm
# STL-transform residuals and erase discriminating downstream flight),
# while the aligned formula + this penalty term scored best of all
# variants (Spearman +0.458, 17/61 hitters in top-30; weights 20-100
# were equivalent, 50 chosen as mid-range).
WALL_PENALTY_MM = 50.0

# The all-ion mean is dominated by the bulk of the beam that splats early
# in the einzel region (~324mm out) -- measured on the first
# splat-objective smoke run, zero-hit configs differed by only ~3.5mm of
# 325mm. Scoring only the closest fraction of ions ("how close does the
# LEADING EDGE of the beam get?") sharpens the gradient: steering even a
# small vanguard of the beam toward the detector moves the objective
# strongly, well before bulk transmission improves. The all-ion mean is
# still stored per-trial in user_attrs["mean_splat_all_mm"].
SPLAT_TOP_FRACTION = 0.10

_DET_LO = np.array([op.DETECTOR_REGION["x"][0], op.DETECTOR_REGION["y"][0],
                    op.DETECTOR_REGION["z"][0]], dtype=float)
_DET_HI = np.array([op.DETECTOR_REGION["x"][1], op.DETECTOR_REGION["y"][1],
                    op.DETECTOR_REGION["z"][1]], dtype=float)


def splat_distances(positions: np.ndarray) -> np.ndarray:
    """
    Per-ion Euclidean distance (mm) from splat position to the detector
    box (axis-aligned point-to-box distance: 0 inside, else distance to
    the nearest face/edge/corner). positions: (N, 3) -> (N,).
    """
    below = np.maximum(_DET_LO - positions, 0.0)
    above = np.maximum(positions - _DET_HI, 0.0)
    return np.linalg.norm(below + above, axis=1)


def mean_splat_distance(positions: np.ndarray, top_fraction: float = 1.0) -> float:
    """
    Mean splat distance over the closest `top_fraction` of ions
    (top_fraction=1.0 -> plain mean over every recorded ion).
    """
    d = splat_distances(positions)
    if top_fraction < 1.0:
        n_keep = max(1, int(np.ceil(len(d) * top_fraction)))
        d = np.sort(d)[:n_keep]
    return float(d.mean())

# Separate from both optimizer.py's beamline_study.db and optimizer_gp.py's
# beamline_study_gp.db -- this study's trial history is a distinct thing
# (RK4-prefiltered, SIMION-confirmed candidates only), shouldn't mix with
# either.
# Estudio nuevo para el objetivo v2 (la escala cambio: J en [0,1]) --
# NUNCA continuar los estudios viejos (escala mm o hits) con este codigo.
STUDY_NAME = "gemelo_v2"
RESULTS_DB = HERE / "studies" / "gemelo_v2.db"

# Bug fix vs optimizer.py's run_simion(): bare "simion" isn't resolvable
# via subprocess(shell=True) in this environment -- needs the qualified
# path (see optimizer_gp.py, found the same issue there first).
SIMION_EXE = op.SIMION_INSTALL_DIR / "simion.exe"
FLY_COMMAND = (
    f'"{SIMION_EXE}" --nogui fly --recording-output=out.txt --programs=0 '
    f'--retain-trajectories=0 --restore-potential=0 "{op.IOB_FILE}"'
)


# ----------------------------------------------------------------------
# Real SIMION calls (the "run SIMION" step)
# ----------------------------------------------------------------------
def run_simion(command: str) -> str:
    result = subprocess.run(
        command, cwd=str(op.SIMION_INSTALL_DIR), shell=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
    )
    return result.stdout


def apply_voltages(chosen: dict) -> None:
    all_volts = {**op.FIXED, **chosen}
    settings = ",".join(f"{n}={v}" for n, v in sorted(all_volts.items()))
    run_simion(f'"{SIMION_EXE}" --nogui fastadj "{op.PA0_FILE}" {settings}')


def run_simion_candidate(voltages: np.ndarray) -> tuple:
    """
    Fly ONE candidate's full voltage array through real SIMION.

    Returns
    -------
    (value, hits, mean_all, features):
        value    -- el objetivo J_v2 en [0,1] (caracterizador.objetivo_v2,
                    con la caracterizacion RICA del recording); BAD_SCORE
                    (=1.0) si SIMION falla o no registra iones.
        hits     -- conteo en la ventana (gate NARROW_AFTER_HITS y reportes).
        mean_all -- distancia media de splat de todo el haz, mm (diagnostico).
        features -- dict completo del caracterizador (se persiste un
                    subconjunto en user_attrs: dataset para la Task B).
    """
    chosen = {k: float(voltages[k - 1]) for k in op.OPTIMIZE}
    try:
        apply_voltages(chosen)
        out = run_simion(FLY_COMMAND)
    except subprocess.CalledProcessError:
        return BAD_SCORE, 0, 1.0e6, {}

    positions = op.get_positions(out)
    if positions.shape[0] == 0:
        return BAD_SCORE, 0, 1.0e6, {}
    try:
        _, features = desde_simion_ultimo_vuelo()   # recording rico (velocidades)
    except Exception:
        features = caracterizar(positions)          # degradado: solo posiciones
    value, _ = objetivo_v2(features, con_pared=False)
    hits = int(features.get("hits") or op.count_hits(positions))
    return float(value), hits, mean_splat_distance(positions), features


# ----------------------------------------------------------------------
# Cheap RK4 batch screening (the "RK4 scoring" step) -- same approach as
# sim_batch.py's score_chunk, kept self-contained here rather than
# imported so this file has everything the loop logic needs in one place.
# ----------------------------------------------------------------------
def sample_voltage_batch(study, m):
    """FIXED electrodes held at their real value; OPTIMIZE ones drawn from
    the study's sampler (informed by every real trial told so far)."""
    sampler = study.sampler
    dist = optuna.distributions.FloatDistribution(low=-1000.0, high=1000.0)
    all_volts = {**op.FIXED, **op.OPTIMIZE}
    batch = np.zeros((m, N_ELECTRODES))
    for i in range(m):
        for key, value in all_volts.items():
            if isinstance(value, float):
                batch[i, key - 1] = value
            else:
                batch[i, key - 1] = sampler.sample_independent(
                    study, trial=None, param_name="x", param_distribution=dist)
    return batch


def sample_voltage_batch_gp_seeded(study, m, n_seeds, perturbation_std, rng_seed=None):
    """
    The GPSampler-appropriate alternative to sample_voltage_batch(): pay
    the GP's real, expensive joint model (sample_relative(), via a REAL
    study.ask() trial) only n_seeds times -- one sample_relative() call
    per seed covers all 8 electrodes at once, since suggest_float caches
    the relative-search-space result across calls on the same trial (see
    conversation for the timing check that confirmed this: ~60ms for the
    first suggest_float call on a fresh trial, ~0.2ms for the rest).

    The remaining (m - n_seeds) candidates are CHEAP local perturbations
    around those seeds (plain numpy Gaussian noise, clipped to bounds, no
    sampler call at all) -- this is what actually lets the GP's
    posterior/uncertainty steer WHERE the cheap RK4-screening batch
    concentrates, rather than every one of the M candidates being uniform
    random noise (which is what sample_independent() would give you with
    GPSampler -- see module docstring).

    Returns
    -------
    voltages_batch : (m, N_ELECTRODES) ndarray
    seed_trial_for_row : dict {row_index: optuna.trial.Trial}
        Which rows of voltages_batch correspond to a REAL asked trial (as
        opposed to a synthetic perturbation) -- the caller needs this to
        know whether to study.tell() that exact trial (if promoted) or
        prune it (if not), versus add_trial() for a promoted perturbation.
    """
    rng = np.random.default_rng(rng_seed)
    n_seeds = max(1, min(n_seeds, m))
    optimize_keys = sorted(op.OPTIMIZE)

    voltages_batch = np.zeros((m, N_ELECTRODES))
    for k, v in op.FIXED.items():
        voltages_batch[:, k - 1] = v

    seed_trial_for_row = {}
    for i in range(n_seeds):
        trial = study.ask()
        for e in optimize_keys:
            low, high = op.OPTIMIZE[e]
            voltages_batch[i, e - 1] = trial.suggest_float(f"V{e}", low, high)
        seed_trial_for_row[i] = trial

    remaining = m - n_seeds
    for j in range(remaining):
        row = n_seeds + j
        base = j % n_seeds  # round-robin: spread perturbations evenly across seeds
        voltages_batch[row] = voltages_batch[base]
        for e in optimize_keys:
            low, high = op.OPTIMIZE[e]
            noisy = voltages_batch[row, e - 1] + rng.normal(0, perturbation_std)
            voltages_batch[row, e - 1] = np.clip(noisy, low, high)

    return voltages_batch, seed_trial_for_row


def rk4_score_chunk(bfm, wall_index, voltages_chunk, start_positions, start_velocities,
                     species, dt, num_steps):
    """
    One BatchRK4Integrator + one BeamProgressScorer call for the whole
    chunk (see sim_batch.py's score_chunk for why one call, not a loop).

    Returns the RK4 PREDICTION OF THE REAL SIMION OBJECTIVE, lower is
    better: best-SPLAT_TOP_FRACTION mean closest-approach distance to the
    detector + HIT_TERM_SCALE_MM * (1 - reached fraction)
    + WALL_PENALTY_MM * wall_hit_fraction.

    Same top-fraction + transmission-term formula as
    run_simion_candidate(), so "RK4 rank 1" means best on the actual
    objective. Walls are a PENALTY, deliberately NOT a termination:
    validate_rk4_filter.py measured terminate_on_wall_hit=True against
    288 archived SIMION outcomes and it lost at every margin tried (the
    approximate kill locations erase the downstream trajectory info that
    actually discriminates configs) -- see WALL_PENALTY_MM's comment.
    """
    n_configs = voltages_chunk.shape[0]
    n = start_positions.shape[0]

    bfm.set_voltages_batch(voltages_chunk)
    beam, config_indices = make_batch_beam(species, start_positions, start_velocities, n_configs)
    trajectory = BatchTrajectory(beam, config_indices)
    BatchRK4Integrator(bfm, config_indices).integrate(trajectory, dt=dt, num_steps=num_steps)

    scorer = BeamProgressScorer(
        bfm=bfm, Trajectory=trajectory, dt=dt, num_steps=num_steps,
        detector_bbox=DETECTOR_BBOX, wall_index=wall_index, wall_hit_margin=1.5,
        wall_check_midpoints=False, wall_check_stride=3,
    )
    result = scorer.combined_score(voltages_chunk, **SCORE_WEIGHTS)

    # SCREENING = el predictor VALIDADO en mm, NO J_v2. Leccion medida el
    # 2026-07-05: se intento usar J_v2 tambien aqui y la campana dio 0
    # hitters en 20 corridas (todas las anteriores: primer hitter en <=7).
    # Causa: J_v2 pesa 0.45 en transmision y 0.40 en forma -- exactamente
    # lo que el RK4 NO sabe medir (no cuenta hits; sus features de forma
    # se contaminan con particulas no terminadas) -- y dejo en 0.15 el
    # unico termino validado (acercamiento, Spearman +0.458 offline). El
    # objetivo que se OPTIMIZA es J_v2 (en SIMION, donde las features son
    # reales); el filtro solo tiene que RANKEAR bien, y esta formula es
    # la que lo hace. Evidencia: playpen/gemelo_v2_intento1_*.db y
    # playpen/analisis_objetivo_v2.txt.
    tdist = result["target_distance"].reshape(n_configs, n)
    reached_fraction = result["reached_target"].reshape(n_configs, n).mean(axis=1)
    wall_fraction = result["hit_wall"].reshape(n_configs, n).mean(axis=1)
    n_keep = max(1, int(np.ceil(n * SPLAT_TOP_FRACTION)))
    top_mean = np.sort(tdist, axis=1)[:, :n_keep].mean(axis=1)
    return (top_mean
            + HIT_TERM_SCALE_MM * (1.0 - reached_fraction)
            + WALL_PENALTY_MM * wall_fraction)


def rk4_score_all(bfm, wall_index, voltages_batch, start_positions, start_velocities,
                   species, dt, num_steps, chunk_size):
    """Chunked wrapper around rk4_score_chunk -- see sim_batch.py's memory note
    for why chunking (not just the batch call above) matters at larger M."""
    M = voltages_batch.shape[0]
    scores = np.empty(M)
    n_chunks = int(np.ceil(M / chunk_size))
    for c in range(n_chunks):
        lo, hi = c * chunk_size, min((c + 1) * chunk_size, M)
        scores[lo:hi] = rk4_score_chunk(bfm, wall_index, voltages_batch[lo:hi],
                                         start_positions, start_velocities, species, dt, num_steps)
    return scores


# ----------------------------------------------------------------------
# The loop itself
# ----------------------------------------------------------------------
def build_study(sampler=None):
    return optuna.create_study(
        direction=DIRECTION,  # "minimize" -- mean splat distance, see OBJECTIVE above
        storage=f"sqlite:///{RESULTS_DB}",
        study_name=STUDY_NAME,
        load_if_exists=True,
        sampler=sampler,  # None -> Optuna's default, TPESampler -- see module docstring
    )


def orchestrate(total_simion_budget, simion_per_iteration, target_score=None,
                 sampler=None, n_gp_seeds=5, perturbation_std=150.0, verbose=True):
    """
    Runs the full GP-suggest -> RK4-screen -> SIMION-confirm -> retrain loop
    until either total_simion_budget real SIMION runs have been spent, or
    (if given) study.best_value reaches target_score.

    Parameters
    ----------
    total_simion_budget : int
        Total real SIMION candidates to run across ALL iterations combined
        -- the actual expensive-resource budget for this whole call.
    simion_per_iteration : int
        How many of the RK4-screened candidates to promote to SIMION each
        iteration (this iteration's K in cost_planner.plan_batch terms).
        M (how many to RK4-screen first) is derived from this automatically.
    target_score : float, optional
        Stop early if study.best_value reaches this (only meaningful with
        OBJECTIVE="hits", where higher is better -- see optimizer.py).
    sampler : optuna.samplers.BaseSampler, optional
        Passed to optuna.create_study(); None uses Optuna's default (TPE).
        If this IS a GPSampler (isinstance check), candidate generation
        automatically switches to sample_voltage_batch_gp_seeded() instead
        of the cheap sample_independent() path -- see module docstring for
        why: GPSampler.sample_independent() ignores the GP entirely, so
        using it with GPSampler would waste the GP's whole point.
    n_gp_seeds, perturbation_std : only used when sampler is a GPSampler.
        n_gp_seeds real study.ask() calls per iteration (paying the GP's
        real, ~100ms-300ms+ sample_relative() cost that many times, not M
        times), then the rest of the RK4-screening batch is cheap Gaussian
        perturbation (std in volts) around those seeds -- see
        sample_voltage_batch_gp_seeded()'s docstring for the full reasoning.

    Returns
    -------
    optuna.Study -- the (persisted) study, so you can inspect
    study.best_trial / study.trials_dataframe() afterward.
    """
    gp_seeded = isinstance(sampler, optuna.samplers.GPSampler)
    op.check_setup()

    print("Loading field map...")
    bfm = BatchBasisFieldMap.from_directory(HERE, n_electrodes=N_ELECTRODES)
    print("Building wall collision index...")
    wall_index = build_wall_index(HERE, exclude=WALL_EXCLUDE, target_spacing=2.0)
    species = IonSpecies(mass=28 * 1.66053906660e-27, charge=1.602176634e-19)

    study = build_study(sampler=sampler)

    simion_spent = 0
    iteration = 0
    while simion_spent < total_simion_budget:
        iteration += 1
        k = min(simion_per_iteration, total_simion_budget - simion_spent)
        plan = plan_batch(simion_budget_k=k)
        M, N, T = plan["M"], plan["rk4_particles"], plan["rk4_steps"]
        chunk_size = min(M, 50)

        print(f"\n=== Iteration {iteration} (SIMION spent so far: {simion_spent}/{total_simion_budget}) ===")
        print(f"  Plan: screen M={M} candidates ({N} particles x {T} steps), promote top K={k} to SIMION")

        start_positions, start_velocities = make_beam(
            N=N, species=species, start_point=[395.0, 75.0, 77.0],
            mean_energy_eV=15.0, std_energy_eV=0.42466, half_angle_deg=15.0, seed=42,
        )

        n_real_trials = len(study.get_trials(deepcopy=False))
        seed_trial_for_row = {}
        if gp_seeded:
            print(f"  Sampling {M} candidates: {n_gp_seeds} real GP-informed seeds (sampler informed by "
                  f"{n_real_trials} real SIMION trials so far) + {M - n_gp_seeds} cheap local perturbations...")
            t0 = time.time()
            voltages_batch, seed_trial_for_row = sample_voltage_batch_gp_seeded(
                study, M, n_gp_seeds, perturbation_std, rng_seed=iteration)
            print(f"    ({len(seed_trial_for_row)} seed(s) sampled in {time.time()-t0:.1f}s)")
        else:
            print(f"  Sampling {M} candidates (sampler informed by {n_real_trials} real SIMION trials so far)...")
            voltages_batch = sample_voltage_batch(study, M)

        # Option #3 (SESSION_HANDOFF.txt section 8): if bender_field_analysis.py
        # has produced a physically-informed starting point, re-center part of
        # the screening batch on it -- enqueue_trial() alone wouldn't reach the
        # TPE path (it never ask()s), see starting_point.py. Once a real trial
        # WITH HITS exists, a second block exploits tightly around it (the RK4
        # proxy only finds the neighborhood; local search around confirmed
        # hits is what climbs -- see starting_point.py BEST_FRACTION note).
        # GP-seeded rows 0..n_seeds-1 are real asked trials and must not be
        # overwritten, so injection goes into the tail block for that path.
        # Seeded by the study's real trial count, NOT the iteration number:
        # iteration restarts at 1 on every orchestrate() call, so seeding by
        # iteration made separate runs regenerate the exact same perturbations
        # and re-spend SIMION budget on already-evaluated candidates (caught
        # live: two runs promoted identical top-10s). n_real_trials grows
        # every iteration, so this stays deterministic but never repeats.
        best_center = None
        best_hits = 0
        best_hits_trial = None
        for t in study.get_trials(deepcopy=False):
            h = t.user_attrs.get("simion_hits")
            if h is not None and h > best_hits and all(f"V{e}" in t.params for e in op.OPTIMIZE):
                best_hits, best_hits_trial = h, t
        if best_hits > NARROW_AFTER_HITS:
            best_center = np.zeros(N_ELECTRODES)
            for e_num, v_fixed in op.FIXED.items():
                best_center[e_num - 1] = v_fixed
            for e_num in op.OPTIMIZE:
                best_center[e_num - 1] = best_hits_trial.params[f"V{e_num}"]
        elif best_hits > 0:
            print(f"  Search staying BROAD: best so far {best_hits:g} hits <= "
                  f"narrow threshold ({NARROW_AFTER_HITS})")
        inject_rng = np.random.default_rng(1000 * n_real_trials + iteration)
        target = voltages_batch[len(seed_trial_for_row):] if gp_seeded else voltages_batch
        n_sp, n_best = inject_starting_point(target, op.OPTIMIZE, rng=inject_rng,
                                             best_center=best_center)
        if n_best:
            print(f"  Injected {n_best} tight perturbations (std 40V) around best SIMION trial "
                  f"({best_hits:g} hits) + {n_sp} rows around the derived starting point")
        elif n_sp:
            print(f"  Injected derived starting point + {n_sp - 1} perturbations "
                  f"around it into the screening batch (see derived_starting_point.json)")

        t0 = time.time()
        scores = rk4_score_all(bfm, wall_index, voltages_batch, start_positions, start_velocities,
                                species, DT, T, chunk_size)
        print(f"  Stage-A RK4 screen: {M} candidates in {time.time()-t0:.1f}s")

        # Stage B: re-fly only the stage-A shortlist at higher fidelity
        # and a different beam seed (see RESCREEN_* constants); promotion
        # ranks on THESE scores. Lower is better throughout now -- the RK4
        # score is a prediction of the minimize-direction SIMION objective.
        n_rescreen = int(min(M, max(3 * k, RESCREEN_MIN)))
        shortlist = np.argsort(scores)[:n_rescreen]
        rescreen_positions, rescreen_velocities = make_beam(
            N=RESCREEN_PARTICLES, species=species, start_point=[395.0, 75.0, 77.0],
            mean_energy_eV=15.0, std_energy_eV=0.42466, half_angle_deg=15.0,
            seed=RESCREEN_SEED)
        t0 = time.time()
        scores_b = rk4_score_all(bfm, wall_index, voltages_batch[shortlist],
                                  rescreen_positions, rescreen_velocities, species,
                                  RESCREEN_DT, RESCREEN_STEPS, RESCREEN_CHUNK)
        print(f"  Stage-B rescreen: top {n_rescreen} at {RESCREEN_PARTICLES} particles "
              f"(seed {RESCREEN_SEED}, dt {RESCREEN_DT:g}) in {time.time()-t0:.1f}s")
        score_by_idx = {int(i): float(s) for i, s in zip(shortlist, scores_b)}
        promotion_order = shortlist[np.argsort(scores_b)]

        # Promote best-first, but never re-spend SIMION budget on a voltage
        # array the study has already evaluated (across ALL past runs -- the
        # study is persistent) or on an exact duplicate within this batch
        # (e.g. the un-perturbed starting-point row is injected every
        # iteration). Keys rounded to 0.1V: below both SIMION's fastadj
        # resolution-of-interest and the perturbation scale.
        optimize_keys = sorted(op.OPTIMIZE)
        seen = {
            tuple(round(t.params[f"V{e}"], 1) for e in optimize_keys)
            for t in study.get_trials(deepcopy=False)
            if all(f"V{e}" in t.params for e in optimize_keys)
        }
        top_k_idx = []
        n_dupes_skipped = 0
        for idx in promotion_order:
            key = tuple(round(float(voltages_batch[idx][e - 1]), 1) for e in optimize_keys)
            if key in seen:
                n_dupes_skipped += 1
                continue
            seen.add(key)
            top_k_idx.append(int(idx))
            if len(top_k_idx) == k:
                break
        top_k_idx = np.array(top_k_idx, dtype=int)
        if n_dupes_skipped:
            print(f"  Skipped {n_dupes_skipped} already-evaluated/duplicate candidate(s) during promotion")
        if len(top_k_idx) == 0:
            print("  Every screened candidate was already evaluated -- stopping to avoid a wasted loop.")
            break
        n_seeds_promoted = sum(1 for idx in top_k_idx if idx in seed_trial_for_row)
        promoted_scores = [score_by_idx[idx] for idx in top_k_idx]
        print(f"  Promoting top {len(top_k_idx)} (score RK4 validado "
              f"{min(promoted_scores):.1f} a {max(promoted_scores):.1f}mm) to real SIMION"
              + (f" ({n_seeds_promoted} of them GP seeds)..." if gp_seeded else "..."))

        dist = optuna.distributions.FloatDistribution(low=-1000.0, high=1000.0)
        top_k_idx_set = set(int(i) for i in top_k_idx)
        for rank, idx in enumerate(top_k_idx, 1):
            voltages = voltages_batch[idx]
            t0 = time.time()
            real_value, simion_hits, mean_all, feats = run_simion_candidate(voltages)
            simion_spent += 1
            elapsed = time.time() - t0
            user_attrs = {
                # Prediccion J_v2 del RK4 (etapa B) -- comparable 1:1 con
                # el value del trial (misma formula, mismo caracterizador).
                "rk4_score": score_by_idx[idx],
                "rk4_stage_a_score": float(scores[idx]),
                "iteration": iteration,
                "rk4_rank_in_iteration": rank,
                "simion_elapsed_s": elapsed,
                "gp_seed": bool(idx in seed_trial_for_row),
                "simion_hits": simion_hits,
                "mean_splat_all_mm": mean_all,
            }
            # features del caracterizador -> dataset de calibracion (Task B)
            for fk in ("dist_punta_mm", "n_plane", "offset_x_mm", "offset_y_mm",
                       "sigma_x_mm", "sigma_y_mm", "halo_fraction", "kurtosis_x",
                       "kurtosis_y", "div_x_mrad", "div_y_mrad", "twiss_alpha_x",
                       "twiss_alpha_y", "emittance_x", "emittance_y",
                       "resid_transporte_x_mm", "resid_transporte_y_mm"):
                if fk in feats:
                    user_attrs[f"f_{fk}"] = float(feats[fk])

            if idx in seed_trial_for_row:
                # Real asked trial -- tell() it directly rather than
                # add_trial()'ing a duplicate; this is the trial the GP's
                # sample_relative() actually produced.
                trial = seed_trial_for_row[idx]
                for name, value in user_attrs.items():
                    trial.set_user_attr(name, value)
                study.tell(trial, real_value)
            else:
                params = {f"V{e}": float(voltages[e - 1]) for e in op.OPTIMIZE}
                distributions = {f"V{e}": dist for e in op.OPTIMIZE}
                trial = optuna.trial.create_trial(
                    params=params, distributions=distributions, value=real_value,
                    user_attrs=user_attrs,
                )
                study.add_trial(trial)

            print(f"    [{rank}/{k}] rk4={score_by_idx[idx]:.1f}mm -> J_real="
                  f"{real_value:.3f}, hits={simion_hits} (haz a {mean_all:.0f}mm)"
                  f"  ({elapsed:.1f}s, total SIMION spent: {simion_spent}/{total_simion_budget})")

        # Seeds that were ask()'d but NOT promoted need to be closed out --
        # otherwise they sit forever as RUNNING trials in the persisted
        # study. PRUNED is the correct state: "we chose not to evaluate
        # this one", not a failure.
        for idx, trial in seed_trial_for_row.items():
            if idx not in top_k_idx_set:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        if gp_seeded:
            n_pruned = len(seed_trial_for_row) - n_seeds_promoted
            if n_pruned:
                print(f"  Pruned {n_pruned} unused GP seed(s) (asked but not RK4-promoted)")

        best = study.best_trial
        print(f"  Best real SIMION result so far: J_v2={best.value:.3f}, "
              f"hits={best.user_attrs.get('simion_hits', '?')} (trial {best.number})")

        # minimize: target_score es el J_v2 considerado suficiente para parar.
        if target_score is not None and best.value <= target_score:
            print(f"\nTarget score {target_score} reached -- stopping early.")
            break

    print(f"\n=== Done: {simion_spent} real SIMION candidates run across {iteration} iteration(s) ===")
    best = study.best_trial
    print(f"Best: J_v2={best.value:.3f}  (hits={best.user_attrs.get('simion_hits', '?')})")
    for name, value in best.params.items():
        print(f"   {name}: {value:.1f}")
    return study


if __name__ == "__main__":
    orchestrate(total_simion_budget=20, simion_per_iteration=10)
