"""
gemelo_GP.py
============

Script to run the GP-seeded optimization run using the starting point.
Loads the digital twin from gemelo.py, configures Optuna's Gaussian Process sampler,
and executes the training loop.
"""

import sys
import pathlib
import optuna

# Import digital twin facade
from gemelo import GemeloDigital

def main():
    print("==================================================")
    print("   Starting GP-Seeded Optimization: gemelo_GP     ")
    print("==================================================")
    
    # 1. Initialize digital twin facade
    # Using study 'gemelo_db_v2' and sqlite database
    tw = GemeloDigital(estudio="gemelo_db_v2", db="studies/gemelo_db_v2.db")
    
    # 2. Check for starting point presence
    sp_path = pathlib.Path("derived_starting_point.json")
    if sp_path.exists():
        print(f"[GP Runner] Found starting point: {sp_path.name}")
        with open(sp_path) as f:
            import json
            voltages = json.load(f)
            print(f"            Voltages: {voltages}")
    else:
        print("[GP Runner] Warning: derived_starting_point.json not found in root.")
        print("            GP will optimize entirely from scratch.")
        
    # 3. Execute training with GPSampler
    print("\n[GP Runner] Configuring GPSampler and launching orchestrator loop...")
    print("            Presupuesto: 60 trials (10 per iteration + 5 GP seeds)...")
    
    try:
        tw.entrenar(
            presupuesto=60,
            por_iteracion=10,
            n_gp_seeds=5,
            sampler=optuna.samplers.GPSampler()
        )
        print("\n==================================================")
        print("           Optimization Finished!                 ")
        print("==================================================")
        
        # 4. Display results
        m = tw.mejor()
        if m:
            print(f"Estudio '{tw.estudio}': {m['n_trials']} trials, mejor = trial {m['trial']} (J_v2={m['objetivo']:.3f}, hits={m['hits']})")
            print(f"  voltajes: {m['voltajes']}")

            print("\nPrediccion del gemelo RK4 para ese config:")
            import numpy as np
            pred = tw.predecir(m["voltajes"])
            print(f"  J_v2 predicho: {pred['objetivo']:.3f} (reach {pred['reach_fraction']*100:.0f}%, wall {pred['wall_fraction']*100:.0f}%)")
            for k, val in pred["features"].items():
                if isinstance(val, (int, np.integer)) or (isinstance(val, float) and not np.isnan(val)):
                    print(f"    {k:16s} = {val:.3f}" if isinstance(val, float) else f"    {k:16s} = {val}")

            if "--evaluar" in sys.argv:
                print("\nEvaluacion SIMION real:")
                ev = tw.evaluar(m["voltajes"])
                print(f"  J_v2 real: {ev['objetivo']:.3f}  hits={ev['hits']}")
                for k, val in ev["features"].items():
                    if isinstance(val, float) and not np.isnan(val):
                        print(f"    {k:16s} = {val:.3f}")
        else:
            print("[GP Runner] No trials completed successfully. Check SIMION logs.")
            
    except Exception as exc:
        print(f"\n[GP Runner] Error during optimization: {exc}")
        sys.exit(1)

if __name__ == "__main__":
    main()
