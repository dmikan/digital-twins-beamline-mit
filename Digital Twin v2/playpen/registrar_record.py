"""Registra el config record (quad x0.70, 22 hits) en el conocimiento del
gemelo: (1) lo vuela via la fachada (queda en el registro jsonl con
features y J v2.1), (2) lo agrega como trial al estudio gemelo_v2 para
que el sampler lo conozca, (3) actualiza derived_starting_point.json para
que la inyeccion del screening se re-centre en la zona nueva."""
import json
import pathlib
import sys

import numpy as np
import optuna

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
optuna.logging.set_verbosity(optuna.logging.WARNING)

from gemelo import GemeloDigital
import optimizer as op

RECORD = {3: -269.6, 6: -208.0, 9: -310.7, 10: 235.3, 11: 260.4, 12: -225.6,
          15: -0.8, 18: -252.6}

tw = GemeloDigital()
print("volando el record en SIMION via la fachada...")
ev = tw.evaluar(RECORD)
print(f"J_v2.1 = {ev['objetivo']:.3f}   hits = {ev['hits']}")
for k, v in sorted(ev["desglose"].items(), key=lambda kv: -kv[1]):
    print(f"  {k:14s} {v:.4f}")

# trial al estudio (para que TPE lo vea)
study = optuna.load_study(study_name=tw.estudio, storage=f"sqlite:///{tw.db}")
dist = optuna.distributions.FloatDistribution(low=-1000.0, high=1000.0)
params = {f"V{e}": float(v) for e, v in RECORD.items()}
attrs = {"simion_hits": ev["hits"], "origen": "scan_combo_cuerpo 2026-07-05"}
for fk, fv in ev["features"].items():
    if isinstance(fv, (int, float)):
        attrs[f"f_{fk}"] = float(fv)
study.add_trial(optuna.trial.create_trial(
    params=params, distributions={k: dist for k in params},
    value=float(ev["objetivo"]), user_attrs=attrs))
print(f"trial agregado al estudio '{tw.estudio}' (ahora {len(study.trials)} trials)")

# nuevo punto inicial fisico (respaldo del anterior)
sp_file = ROOT / "derived_starting_point.json"
respaldo = ROOT / "playpen" / "derived_starting_point_v1_respaldo.json"
respaldo.write_text(sp_file.read_text())
sp_file.write_text(json.dumps({f"V{e}": v for e, v in sorted(RECORD.items())}, indent=2))
print(f"derived_starting_point.json actualizado (respaldo en playpen/)")
