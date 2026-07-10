"""
run_tpe.py
==========
Runs a fresh TPE optimization study on the full 8D search space for exactly 60
SIMION evaluations (matching the 61 total trials budget of the other studies)
to allow a fair convergence and performance comparison.
"""
import pathlib
import sys
import optuna

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import optimizer as op

STUDY_NAME = "gemelo_v2"
DB_PATH = ROOT / "studies" / "gemelo_v2.db"
BACKUP_PATH = ROOT / "studies" / "gemelo_v2_100runs.db"

def main():
    # 1. Backup legacy 100-trial DB if it exists and hasn't been backed up yet
    if DB_PATH.exists():
        try:
            # Check trial count
            study = optuna.load_study(study_name=STUDY_NAME, storage=f"sqlite:///{DB_PATH}")
            n_trials = len(study.trials)
            print(f"Base de datos actual '{DB_PATH.name}' tiene {n_trials} trials.")
            if n_trials > 70:
                if not BACKUP_PATH.exists():
                    DB_PATH.rename(BACKUP_PATH)
                    print(f"Resguardada base de datos de 100 trials en '{BACKUP_PATH.name}'.")
                else:
                    DB_PATH.unlink()
                    print(f"Eliminada base de datos legacy (backup ya existe en '{BACKUP_PATH.name}').")
            else:
                print(f"La base de datos actual ya tiene un presupuesto reducido ({n_trials} trials). No se requiere backup.")
        except Exception as e:
            print(f"Error al verificar base de datos existente: {e}")

    # 2. Run TPE optimization on 8D space with 60 budget
    print(f"Iniciando optimizacion TPE (8D) para {STUDY_NAME} con presupuesto de 60 SIMION runs...")
    op.orchestrate(
        total_simion_budget=60,
        simion_per_iteration=10,
        n_gp_seeds=5,
        perturbation_std=150.0,
        studyname=STUDY_NAME,
        db_path=DB_PATH,
        reduced_search=False,
        sampler=optuna.samplers.TPESampler(),
    )
    print("Optimizacion TPE finalizada.")

    # 3. Regenerate figures
    print("Regenerando figuras cientificas...")
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "tools" / "report_figures.py")], check=True)
    print("Figuras regeneradas con exito.")

if __name__ == "__main__":
    main()
