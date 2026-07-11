import pathlib
import sys
import numpy as np
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import optimizer as op

def main():
    # 1. Load trials from gemelo_v2 and gemelo_db_v2
    studies = ["gemelo_v2", "gemelo_db_v2"]
    pts = []

    for study_name in studies:
        db_path = ROOT / "studies" / f"{study_name}.db"
        if not db_path.exists():
            continue
        try:
            study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")
            for t in study.trials:
                if t.value is not None:
                    hits = t.user_attrs.get("simion_hits")
                    if hits is None:
                        hits = t.value if (t.value >= 0) else None
                    if hits is not None and hits > 0:
                        # Check if all bender voltages are present in params
                        if all(f"V{e}" in t.params for e in (9, 10, 11, 12)):
                            pts.append({
                                "V9": t.params["V9"],
                                "V10": t.params["V10"],
                                "V11": t.params["V11"],
                                "V12": t.params["V12"],
                                "hits": hits
                            })
        except Exception as e:
            print(f"Error cargando {study_name}: {e}")

    if not pts:
        print("No hay suficientes datos con hits > 0 para realizar analisis.")
        sys.exit(0)

    # 2. Extract voltages
    V9 = np.array([p["V9"] for p in pts])
    V10 = np.array([p["V10"] for p in pts])
    V12 = np.array([p["V12"] for p in pts])
    hits = np.array([p["hits"] for p in pts])

    # 3. Create 3D plot of V9 vs V10 vs V12 vs target hits
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of V9 vs V10 vs V12, colored by Hits
    sc = ax.scatter(V9, V10, V12, c=hits, cmap="plasma", edgecolor='k', lw=0.4, s=60, alpha=0.85)
    fig.colorbar(sc, ax=ax, label="SIMION Hits (de 500)", shrink=0.6, aspect=12)

    # Labeling (without accents for Julian's voice)
    ax.set_xlabel("Voltaje de electrodo V9 (V)", fontweight='bold')
    ax.set_ylabel("Voltaje de electrodo V10 (V)", fontweight='bold')
    ax.set_zlabel("Voltaje de electrodo V12 (V)", fontweight='bold')
    ax.set_title("Espacio de Voltajes de Electrodos Bender: V9 vs V10 vs V12\n"
                 "Los puntos estan coloreados segun la cantidad de hits lograda")

    # Set standard view
    ax.view_init(elev=25, azim=45)

    out_dir = ROOT / "outputs" / "report_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_electrodos_bender_3d.png", dpi=150)
    plt.close(fig)
    print(f"\nok  fig_electrodos_bender_3d.png guardado en {out_dir}")

if __name__ == "__main__":
    main()
