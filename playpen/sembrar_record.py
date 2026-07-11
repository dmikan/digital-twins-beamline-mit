"""
sembrar_record.py
==================

Crea el "gemelo que arranca del record": deja el estudio gemelo_v2 y el
punto de partida del orchestrator apuntando a la cadena de colimacion
confirmada (V3=250, quad x0.70, V6=-750 -- 43 hits, 2026-07-06).

  1. sembrar_desde_registro(): TODOS los vuelos SIMION reales de hoy
     entran al estudio como trials (los buenos marcan el tesoro, los
     muertos ensenan que evitar).
  2. derived_starting_point.json <- config record (respaldo del anterior
     en playpen/derived_starting_point_prev.json): cada batch de
     screening del orchestrator re-centra ~40% de candidatos ahi.
  3. Verifica tw.mejor() y mide la prediccion RK4 del record (para saber
     cuanto lo castiga el punto ciego del filtro).

Correr:  python playpen/sembrar_record.py
"""

import json
import pathlib
import shutil
import sys

import numpy as np
import optuna

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

optuna.logging.set_verbosity(optuna.logging.WARNING)

from gemelo import GemeloDigital
from validate_rk4_filter import collect_archived_trials

QUAD = (9, 10, 11, 12)
SP_FILE = ROOT / "derived_starting_point.json"
SP_BACKUP = ROOT / "playpen" / "derived_starting_point_prev.json"


def main():
    # ---- el config record: la fila SIMION con MAS HITS del registro ----
    # (v2 2026-07-06: antes se derivaba de collect_archived_trials, pero
    # esa recoleccion fluctua con OneDrive y una vez devolvio otra "base"
    # -- el registro es la fuente de verdad estable de lo que YA medimos)
    tw = GemeloDigital()
    corridas = [f for f in tw.historial(fuente="simion") if f.get("voltajes")]
    if not corridas:
        raise SystemExit("Registro sin corridas SIMION -- nada que sembrar.")
    mejor_fila = max(corridas, key=lambda f: f.get("hits") or 0)
    record = tw.voltajes_completos(mejor_fila["voltajes"])
    rec_dict = {f"V{e}": round(float(record[e - 1]), 1)
                for e in (3, 6, 9, 10, 11, 12, 15, 18)}
    print(f"Config record ({mejor_fila.get('hits')} hits, "
          f"J={mejor_fila.get('objetivo'):.3f}): {rec_dict}")

    # ---- 1. sembrar el estudio con todo el registro SIMION ----
    tw.sembrar_desde_registro()

    # ---- 2. punto de partida del orchestrator = record ----
    if SP_FILE.exists():
        shutil.copy(SP_FILE, SP_BACKUP)
        print(f"Respaldo del punto de partida anterior: {SP_BACKUP.name}")
    SP_FILE.write_text(json.dumps(rec_dict, indent=2), encoding="utf-8")
    print(f"derived_starting_point.json <- config record")

    # ---- 3. verificar ----
    m = tw.mejor()
    print(f"\nEstudio '{tw.estudio}': {m['n_trials']} trials")
    print(f"  mejor: trial {m['trial']}, J_v2={m['objetivo']:.3f}, hits={m['hits']}")
    print(f"  voltajes: {m['voltajes']}")

    print("\nPrediccion RK4 del record (cuantifica el punto ciego del filtro):")
    pred = tw.predecir(record)
    print(f"  J_v2 predicho: {pred['objetivo']:.3f} (real medido: ~0.56)  "
          f"reach={pred['reach_fraction']:.2f}  wall={pred['wall_fraction']:.2f}")


if __name__ == "__main__":
    main()
