# 🔬 Digital Twin of an Ion Beamline

> A physics-informed digital twin for optimizing a 90° ion bending beamline using Gaussian Processes, RK4 simulation, and SIMION as ground truth.

---

## Overview

This project implements a **digital twin** of an electrostatic ion beamline that bends a Si⁺ ion beam 90° from source to detector. The system uses a two-stage optimization loop that combines a fast physics-based RK4 surrogate model with high-fidelity SIMION simulations, guided by Bayesian optimization (Gaussian Processes via Optuna).

The beamline consists of **19 electrodes**. The digital twin optimizes **8 free electrode voltages** (electrodes 3, 6, 9–12, 15, 18; range ±1000 V) to maximize ion transmission to the detector, while keeping the rest fixed (source: +500 V, detector: −2000 V, grounds).

**Ion beam parameters:**
- Species: Si⁺ (28 amu, 15 eV)
- Initial beam: 500 ions in a 15° cone from position (395, 75, 77) mm
- Target detector region: x ∈ [70–82] mm, y ∈ [70–83] mm, z ∈ [403–407] mm

---

## Key Results

| Metric | Value |
|---|---|
| RK4 evaluation speed | ~0.25 s/candidate |
| SIMION evaluation speed | ~5.77 s/candidate |
| Speed ratio (RK4 vs SIMION) | **~23×** faster |
| RK4 filter enrichment over random | **2.7×** |
| Hitter zone fraction of search space | ~1 × 10⁻⁹ |
| Best known configuration | 7d bender parametrization (Search v2.1) |

---

## Architecture

```
Optimizer (Optuna TPE / GP)
        │
        ▼
 [Stage A – Cheap Screening]
  RK4 Physics Engine
  239 candidates × 50 particles × 1500 steps
        │  top candidates
        ▼
 [Stage B – Fine Screening]
  RK4 Physics Engine
  top-K × 200 particles × 3000 steps
        │  top-10
        ▼
 [Ground Truth Evaluation]
  SIMION 8.1
        │  results feed back
        └──────────────────────► Optuna DB (studies/)
```

The **RK4 engine** exploits the linearity of Laplace's equation: the electric field for any voltage configuration is computed as a linear superposition of 19 precomputed basis electrode fields, loaded once at startup. Trajectories are integrated in a vectorized batch (all candidates and particles simultaneously) using NumPy.

---

## Project Structure

```
.
├── gemelo_GP.py            # Trainable GP digital twin (main entry point for ML)
├── caracterizador.py       # Beam characterizer: computes J_v2.4 objective
├── physics.py              # RK4 integrator, composite grid, wall collision
├── optimizer.py            # SIMION interface, electrode definitions (course baseline)
├── basis_electrode_*.csv   # 19 precomputed basis electric field maps
├── SimpleSetUp.*           # SIMION setup files (.iob, .fly2, .rec, .lua)
├── derived_starting_point.json  # Physics-derived initial guess
│
├── tools/                  # Analysis and utility scripts
│   ├── fresh_run.py        # Reproducible clean benchmark run
│   ├── validate_rk4_filter.py   # Offline RK4 filter validation
│   ├── bender_field_analysis.py # Quadrupole field analysis & starting point
│   ├── sim_batch.py        # Standalone RK4 screening (no SIMION)
│   ├── steer_scan.py       # Steering electrode scan
│   ├── plot_rk4_trajectories.py # 3D trajectory visualization
│   ├── report_figures.py   # Publication-quality figures
│   ├── generate_report_pdf.py   # PDF report generator
│   └── especificacion_gemelo_digital.md  # Full technical specification
│
├── studies/                # Optuna SQLite databases (active studies)
│   ├── gemelo_db_v2.db
│   ├── gemelo_v2_bender6d.db
│   └── gemelo_v2_bender7d.db
│
├── outputs/                # All figures, CSVs, and reports go here
├── docs/                   # Guides, calculations, and field notes
│   ├── GUIA_CODIGO.txt     # Detailed code guide
│   └── CALCULOS_INFORME.txt  # Supporting calculations for the report
├── playpen/                # Scratch area (not used by the twin)
└── legacy/                 # Archived studies and superseded versions
```

---

## Quick Start

### Prerequisites

```bash
pip install numpy scipy optuna trimesh matplotlib
# For GP sampler (optional):
# pip install torch
# export KMP_DUPLICATE_LIB_OK=TRUE
```

You also need:
- **SIMION 8.1** installed (set `SIMION_INSTALL_DIR` in `optimizer.py`)
- `basis_electrode_1..19.csv` — precomputed basis field maps (included)
- `SimpleSetUp.iob` and `electrode_.PA0` — SIMION setup files (included)

### Using the Digital Twin Facade

```python
from gemelo_GP import GemeloEntrenable

tw = GemeloEntrenable()

# Inspect current best configuration
print(tw.resumen())
print(tw.mejor())

# Predict with RK4 physics model (~seconds)
tw.predecir(voltajes)

# Predict with GP statistical model (instant, with uncertainty)
tw.predecir_modelo(voltajes)

# Evaluate with real SIMION run (~6 s)
tw.evaluar(voltajes)

# Train the GP surrogate on historical data
tw.entrenar_modelo()

# Suggest new candidates via Expected Improvement
for candidate in tw.sugerir(3):
    print(candidate)

# Run a full optimization cycle (uses SIMION budget)
tw.ciclo(rondas=1, k=2)
```

### Running the Optimization Loop

```bash
# Clean benchmark run — 50 SIMION evaluations, fresh study
python tools/fresh_run.py 50

# Continue an existing study
python tools/fresh_run.py 20 --continue

# Validate RK4 filter offline (no SIMION cost)
python tools/validate_rk4_filter.py

# Regenerate the physics-derived starting point
python tools/bender_field_analysis.py
```

---

## The Objective Function: J_v2.4

The system minimizes a scalar objective $J_{v2.4} \in [0, 1]$ combining multiple beam quality metrics into a single value (lower = better):

$$J_{v2.4} = \text{normalize}\left[\alpha \cdot d_{\text{top10\%}} + \lambda\!\left(1 - \frac{\text{hits}}{N_{\text{ref}}}\right) + \text{terms}(\sigma_x, \sigma_y, \text{halo}, \text{kurtosis}, \ldots)\right]$$

| Term | Description |
|---|---|
| $d_{\text{top10\%}}$ | Mean distance of the closest 10% of ions to the detector |
| $\lambda = 2$ | Linear transmission weight (prevents saturation at high hit counts) |
| $\sigma_x, \sigma_y$ | RMS beam size |
| halo fraction | Fraction of particles outside the beam core |
| kurtosis | Distribution sharpness |
| angular divergence | Beam opening angle |
| Twiss emittance | Phase-space beam quality |

---

## Quadrupole Reparametrization

To reduce the effective search dimensionality, the four quadrupole electrode voltages (V₉–V₁₂) are encoded using three physical parameters:

| Parameter | Physical meaning |
|---|---|
| **A** | Common voltage offset |
| **B** | Main bending strength |
| **C** | Horizontal asymmetry |

$$V_9 = A + B + C \quad V_{10} = A - B \quad V_{11} = A - B \quad V_{12} = A + B - C$$

This encoding preserves the physical symmetry of the quadrupole while reducing the dimensionality of the optimization problem.

---

## RK4 Physics Engine

The fast surrogate integrates ion trajectories using a 4th-order Runge-Kutta integrator:

- **Time step:** $dt = 10^{-8}$ s → ~0.6 mm per step
- **Basis superposition:** field = linear combination of 19 precomputed basis fields
- **Composite grid:** high-resolution local boxes (1.0 mm quadrupole, 2.5 mm collimators) over a coarser global grid (2.0 mm) → **15× memory reduction**
- **Wall collision:** metal voxels extracted from SIMION's `.PA0`, queried via spatial tree every 3 integration steps (4.2× speedup over every-step checking)

---

## Reproducibility Notes

- Stage A beam seed: **42** | Stage B beam seed: **1234**
- SIMION's ion source is not seeded → hit counts may fluctuate slightly between identical runs
- `fresh_run.py` refuses to overwrite an existing study DB, guaranteeing true fresh starts
- Perturbation seeds are deterministic: `seed = 1000 × n_trials + iteration`

---

## Documentation

| File | Contents |
|---|---|
| [`docs/GUIA_CODIGO.txt`](docs/GUIA_CODIGO.txt) | Full code guide: what each file does, how to run each workflow, tunable parameters |
| [`docs/CALCULOS_INFORME.txt`](docs/CALCULOS_INFORME.txt) | Supporting calculations cited in the technical report |
| [`tools/especificacion_gemelo_digital.md`](tools/especificacion_gemelo_digital.md) | Full technical specification: architecture, hyperparameters, validation methodology |

---

## License

This project was developed as part of a research collaboration. Please contact the authors before reusing or redistributing.

---

*Developed at MIT · 2026*
