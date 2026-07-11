import optuna
import glob
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)

def main():
    db_paths = glob.glob(str(ROOT / "studies" / "*.db"))
    
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    colors = {
        "gemelo_db_v2": "#9b59b6",      # Purple
        "gemelo_v2_bender6d": "#3498db", # Blue
        "gemelo_v2_bender7d": "#e67e22", # Orange
    }
    
    labels = {
        "gemelo_db_v2": "8D Full Space (gemelo_db_v2)",
        "gemelo_v2_bender6d": "6D/7D Extremo (gemelo_v2_bender6d)",
        "gemelo_v2_bender7d": "7D Asimétrico Suave (gemelo_v2_bender7d)",
    }
    
    for db in sorted(db_paths):
        name = pathlib.Path(db).stem
        
        try:
            summaries = optuna.study.get_all_study_summaries(storage=f"sqlite:///{db}")
        except Exception:
            continue
            
        for s in summaries:
            try:
                study = optuna.load_study(study_name=s.study_name, storage=f"sqlite:///{db}")
            except Exception:
                continue
                
            completed_trials = sorted(
                [t for t in study.trials if t.value is not None and t.state == optuna.trial.TrialState.COMPLETE],
                key=lambda t: t.number
            )
            if not completed_trials:
                continue
                
            vals = [t.value for t in completed_trials]
            running_min = np.minimum.accumulate(vals)
            
            col = colors.get(name, "gray")
            lbl = labels.get(name, name)
            
            ax.plot(range(1, len(running_min) + 1), running_min, marker="o", markersize=4, 
                    linestyle="-", color=col, label=lbl, alpha=0.85, lw=1.5)
                    
    ax.set_xlabel("Número de Evaluación en SIMION")
    ax.set_ylabel("Costo Mínimo Acumulado J (adimensional, menor = mejor)")
    ax.set_title("Comparativa de Convergencia: Modelos Activos en el Workspace\nMuestra la optimización desde voltajes en 0V en limpio", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0.55, 1.15)
    
    fig.tight_layout()
    fig.savefig(OUT / "fig_comparativa_v2_series.png", dpi=150)
    plt.close(fig)
    print("ok  fig_comparativa_v2_series.png")

if __name__ == "__main__":
    main()
