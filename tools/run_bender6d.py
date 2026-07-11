"""
run_bender6d.py
===============

Runner for the reduced 6D bender reparametrization (see
docs/INSTRUCCIONES_BENDER_6D.md): the sampler searches {V3, V6, A, B,
V15, V18} instead of the raw 8 electrodes; A/B expand linearly to
V9..V12 (fixed map in optimizer.py: expand_bender). physics.py/SIMION
are unaffected -- they always receive the expanded full 19-electrode
vector, same as the existing 8D loop.

Uses a DEDICATED study/db (gemelo_v2_bender6d) so it never mixes with
the existing gemelo_v2 / gemelo_db_v2 8D studies (different Optuna
search space -- mixing would confuse TPE/GP's model of the space).

Run:
    python tools/run_bender6d.py
"""
import pathlib
import sys
import optuna

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import optimizer as op

STUDY_NAME = "gemelo_v2_bender6d"
DB_PATH = ROOT / "studies" / "gemelo_v2_bender6d.db"

# Matched against gemelo_db_v2's historical GP-seeded budget (60 total,
# 10/iteration, 5 GP seeds) for a fair comparison: same sampler class,
# same budget, only the search space (8D vs this 6D reparametrization)
# differs.
if __name__ == "__main__":
    op.orchestrate(
        total_simion_budget=60,
        simion_per_iteration=10,
        n_gp_seeds=5,
        perturbation_std=150.0,
        studyname=STUDY_NAME,
        db_path=DB_PATH,
        reduced_search=True,
        sampler=optuna.samplers.GPSampler(),
    )
