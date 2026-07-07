"""
gemelo.py
=========

LA PUERTA DE ENTRADA UNICA AL GEMELO DIGITAL (Consolidated).

Wraps the physical system, characterizer, and optimizer behind a single facade.
All imports are consolidated to the three main root modules:
  - physics.py (phys)
  - caracterizador.py (carac)
  - optimizer.py (op)
"""

import datetime
import json
import pathlib
import sys
import optuna
import numpy as np

import physics as phys
import optimizer as op
import caracterizador as carac

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE

# Register all runs made via the facade:
REGISTRO = ROOT / "studies" / "registro_corridas.jsonl"

class GemeloDigital:
    """Facade for the digital twin: physics loaded once, persistent study."""

    def __init__(self, estudio=None, db=None):
        self.estudio = estudio or op.STUDY_NAME
        self.db = pathlib.Path(db) if db else op.RESULTS_DB
        self._bfm = None
        self._wall = None
        self._margen = op.WALL_HIT_MARGIN
        self._species = phys.IonSpecies(mass=28 * 1.66053906660e-27,
                                   charge=1.602176634e-19)

    @property
    def fisica(self):
        if self._bfm is None:
            # cargar_fisica automatically detects Campo1mm or coarse fallback
            fis = phys.cargar_fisica(ROOT)
            self._bfm = fis.bfm
            self._wall = fis.wall
            self._margen = fis.margen
        return self._bfm, self._wall

    def voltajes_completos(self, voltajes):
        v = np.zeros(19)
        for e, val in op.FIXED.items():
            v[e - 1] = val
        if isinstance(voltajes, dict):
            for e, val in voltajes.items():
                e = int(str(e).lstrip("V"))
                if e in op.FIXED:
                    continue
                v[e - 1] = float(val)
        else:
            arr = np.asarray(voltajes, dtype=float)
            for e in op.OPTIMIZE:
                v[e - 1] = arr[e - 1]
        return v

    def predecir(self, voltajes, particulas=200, pasos=None, seed=1234):
        bfm, wall = self.fisica
        v = self.voltajes_completos(voltajes)
        pasos = pasos or op.RESCREEN_STEPS
        dt = op.RESCREEN_DT

        pos0, vel0 = carac.make_beam(N=particulas, species=self._species,
                               start_point=[395.0, 75.0, 77.0], mean_energy_eV=15.0,
                               std_energy_eV=0.42466, half_angle_deg=15.0, seed=seed)
        bfm.set_voltages_batch(v[None, :])
        beam, ci = phys.make_batch_beam(self._species, pos0, vel0, 1)
        tray = phys.BatchTrajectory(beam, ci)
        phys.BatchRK4Integrator(bfm, ci).integrate(tray, dt=dt, num_steps=pasos)

        scorer = carac.BeamProgressScorer(
            bfm=bfm, Trajectory=tray, dt=dt, num_steps=pasos,
            detector_bbox=op.DETECTOR_BBOX, wall_index=wall, wall_hit_margin=self._margen,
            wall_check_midpoints=False, wall_check_stride=3,
        )
        r = scorer.combined_score(v[None, :], **op.SCORE_WEIGHTS)

        tdist = r["target_distance"]
        reached = float(r["reached_target"].mean())
        wall_frac = float(r["hit_wall"].mean())
        n_keep = max(1, int(np.ceil(particulas * op.SPLAT_TOP_FRACTION)))

        posiciones = r["positions"]
        stop = r["stop_idx"]
        idx = np.arange(posiciones.shape[1])
        vels = np.array([s.velocity for s in tray.states])
        limpias = ~r["hit_wall"] & ~r["lost"]
        features = carac.caracterizar(
            posiciones[stop, idx], vels[stop, idx], pos0, vel0,
            mascara=limpias if limpias.sum() >= 3 else None,
            flags=dict(reached=r["reached_target"], lost=r["lost"],
                       hit_wall=r["hit_wall"]),
        )
        features["n_limpias"] = int(limpias.sum())
        features["dist_punta_mm"] = float(np.sort(tdist)[:n_keep].mean())
        objetivo, desglose = carac.objetivo_v2(features)

        resultado = dict(fuente="rk4", objetivo=objetivo, desglose=desglose,
                          reach_fraction=reached, wall_fraction=wall_frac,
                          features=features,
                          voltajes={e: round(float(v[e - 1]), 1) for e in sorted(op.OPTIMIZE)})
        self._registrar(resultado)
        return resultado

    def evaluar(self, voltajes):
        v = self.voltajes_completos(voltajes)
        chosen = {e: float(v[e - 1]) for e in op.OPTIMIZE}
        op.apply_voltages(chosen)
        out = op.run_simion(op.FLY_COMMAND)
        posiciones = op.get_positions(out)
        print(f"[gemelo] SIMION: {posiciones.shape[0]} particulas resueltas")
        if posiciones.shape[0] == 0:
            return dict(fuente="simion", objetivo=op.BAD_SCORE, hits=0, features=None)

        hits = int(op.count_hits(posiciones))
        try:
            _, features = carac.desde_simion_ultimo_vuelo()
        except Exception:
            features = carac.caracterizar(posiciones)
        objetivo, desglose = carac.objetivo_v2(features, con_pared=False)
        resultado = dict(fuente="simion", objetivo=float(objetivo),
                         desglose=desglose, hits=hits,
                         mean_splat_all=op.mean_splat_distance(posiciones),
                         features=features, voltajes=chosen)
        self._registrar(resultado)
        return resultado

    def _registrar(self, resultado):
        try:
            fila = dict(fecha=datetime.datetime.now().isoformat(timespec="seconds"),
                        estudio=self.estudio, **resultado)
            REGISTRO.parent.mkdir(exist_ok=True)
            with open(REGISTRO, "a", encoding="utf-8") as f:
                f.write(json.dumps(fila, default=float) + "\n")
        except Exception as exc:
            print(f"[gemelo] aviso: no se pudo registrar la corrida ({exc})")

    def historial(self, fuente=None, ultimos=None):
        if not REGISTRO.exists():
            return []
        filas = []
        for linea in REGISTRO.read_text(encoding="utf-8").splitlines():
            if not linea.strip():
                continue
            fila = json.loads(linea)
            if fuente is None or fila.get("fuente") == fuente:
                filas.append(fila)
        return filas[-ultimos:] if ultimos else filas

    _FEATURES_SEMILLA = ("dist_punta_mm", "n_plane", "offset_x_mm", "offset_y_mm",
                         "sigma_x_mm", "sigma_y_mm", "halo_fraction", "kurtosis_x",
                         "kurtosis_y", "div_x_mrad", "div_y_mrad", "twiss_alpha_x",
                         "twiss_alpha_y", "emittance_x", "emittance_y",
                         "resid_transporte_x_mm", "resid_transporte_y_mm")

    def _claves_existentes(self, study):
        optimizables = sorted(op.OPTIMIZE)
        return {
            tuple(round(t.params[f"V{e}"], 1) for e in optimizables)
            for t in study.get_trials(deepcopy=False)
            if all(f"V{e}" in t.params for e in optimizables)
        }

    def sembrar(self, voltajes, objetivo, hits=None, features=None):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction=op.DIRECTION, storage=f"sqlite:///{self.db}",
                                    study_name=self.estudio, load_if_exists=True)
        v = self.voltajes_completos(voltajes)
        optimizables = sorted(op.OPTIMIZE)
        clave = tuple(round(float(v[e - 1]), 1) for e in optimizables)
        if clave in self._claves_existentes(study):
            return False
        user_attrs = {"sembrado": True}
        if hits is not None:
            user_attrs["simion_hits"] = int(hits)
        for fk in self._FEATURES_SEMILLA:
            val = (features or {}).get(fk)
            if val is not None and np.isfinite(val):
                user_attrs[f"f_{fk}"] = float(val)
        dist = optuna.distributions.FloatDistribution(low=-1000.0, high=1000.0)
        study.add_trial(optuna.trial.create_trial(
            params={f"V{e}": float(v[e - 1]) for e in op.OPTIMIZE},
            distributions={f"V{e}": dist for e in op.OPTIMIZE},
            value=float(objetivo), user_attrs=user_attrs))
        return True

    def sembrar_desde_registro(self):
        sembradas = 0
        for fila in self.historial(fuente="simion"):
            volts = fila.get("voltajes")
            feats = fila.get("features")
            if feats:
                objetivo = carac.objetivo_v2(feats, con_pared=False)[0]
            else:
                objetivo = fila.get("objetivo")
            if not volts or objetivo is None:
                continue
            if self.sembrar(volts, objetivo, hits=fila.get("hits"), features=feats):
                sembradas += 1
        print(f"[gemelo] {sembradas} corridas SIMION del registro sembradas en '{self.estudio}'")
        return sembradas

    def entrenar(self, presupuesto=20, por_iteracion=10, **kwargs):
        return op.orchestrate(total_simion_budget=presupuesto,
                               simion_per_iteration=por_iteracion,
                               studyname=self.estudio, db_path=self.db, **kwargs)

    def mejor(self, por="objetivo"):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if not self.db.exists():
            return None
        try:
            study = optuna.load_study(study_name=self.estudio, storage=f"sqlite:///{self.db}")
        except KeyError:
            return None
        done = [t for t in study.trials if t.value is not None]
        if not done:
            return None
        if por == "hits":
            con_hits = [t for t in done if t.user_attrs.get("simion_hits") is not None]
            if not con_hits:
                return None
            best = max(con_hits, key=lambda t: t.user_attrs["simion_hits"])
        elif study.direction == optuna.study.StudyDirection.MAXIMIZE:
            best = max(done, key=lambda t: t.value)
        else:
            best = min(done, key=lambda t: t.value)
        return dict(trial=best.number, objetivo=best.value,
                    hits=best.user_attrs.get("simion_hits"),
                    voltajes={k: round(vv, 1) for k, vv in sorted(
                        best.params.items(), key=lambda kv: int(kv[0][1:]))},
                    n_trials=len(done))

    def mejor_global(self):
        import glob
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        mejor = None
        dbs = sorted(glob.glob(str(ROOT / "studies" / "beamline_study_*.db"))) + \
              sorted(glob.glob(str(ROOT / "legacy" / "studies" / "beamline_study_*.db")))
        for db in dbs:
            try:
                summaries = optuna.study.get_all_study_summaries(storage=f"sqlite:///{db}")
            except Exception:
                continue
            for s in summaries:
                study = optuna.load_study(study_name=s.study_name, storage=f"sqlite:///{db}")
                maximize = study.direction == optuna.study.StudyDirection.MAXIMIZE
                for t in study.trials:
                    if t.value is None or not all(f"V{e}" in t.params for e in op.OPTIMIZE):
                        continue
                    h = t.user_attrs.get("simion_hits")
                    if h is None and maximize and t.value >= 0:
                        h = t.value
                    if h is None:
                        continue
                    if mejor is None or h > mejor["hits"]:
                        mejor = dict(hits=float(h), trial=t.number, db=pathlib.Path(db).name,
                                     estudio=s.study_name,
                                     voltajes={k: round(vv, 1) for k, vv in sorted(
                                         t.params.items(), key=lambda kv: int(kv[0][1:]))})
        return mejor

if __name__ == "__main__":
    tw = GemeloDigital()
    m = tw.mejor()
    if m is None:
        print(f"Estudio '{tw.estudio}' vacio -- creando la PRIMERA VERSION del modelo v2...")
        tw.entrenar(presupuesto=50, por_iteracion=10)
        m = tw.mejor()
        if m is None:
            print("El entrenamiento no registro trials -- revisar SIMION.")
            sys.exit(1)
    print(f"Estudio '{tw.estudio}': {m['n_trials']} trials, mejor = trial {m['trial']} (J_v2={m['objetivo']:.3f}, hits={m['hits']})")
    print(f"  voltajes: {m['voltajes']}")

    print("\nPrediccion del gemelo RK4 para ese config:")
    pred = tw.predecir(m["voltajes"])
    print(f"  J_v2 predicho: {pred['objetivo']:.3f} (reach {pred['reach_fraction']*100:.0f}%, wall {pred['wall_fraction']*100:.0f}%)")
    for k, val in pred["features"].items():
        if isinstance(val, (int, np.integer)) or (isinstance(val, float) and not np.isnan(val)):
            print(f"    {k:16s} = {val:.3f}" if isinstance(val, float) else f"    {k:16s} = {val}")

    # tw.entrenar(presupuesto=50, por_iteracion=10)  # force update of study with new prediction
    # print("\nPrediccion del gemelo RK4 para ese config:")
    # pred = tw.predecir(m["voltajes"])
    # print(f"  J_v2 predicho: {pred['objetivo']:.3f} (reach {pred['reach_fraction']*100:.0f}%, wall {pred['wall_fraction']*100:.0f}%)")
    # for k, val in pred["features"].items():
    #     if isinstance(val, (int, np.integer)) or (isinstance(val, float) and not np.isnan(val)):
    #         print(f"    {k:16s} = {val:.3f}" if isinstance(val, float) else f"    {k:16s} = {val}")


        if "--evaluar" in sys.argv:
            print("\nEvaluacion SIMION real:")
            ev = tw.evaluar(m["voltajes"])
            print(f"  J_v2 real: {ev['objetivo']:.3f}  hits={ev['hits']}")
            for k, val in ev["features"].items():
                if isinstance(val, float) and not np.isnan(val):
                    print(f"    {k:16s} = {val:.3f}")
