"""
gemelo.py
==========

LA PUERTA DE ENTRADA UNICA AL GEMELO DIGITAL.

Envuelve las cuatro piezas del sistema detras de una sola clase, cargando
los datos pesados (19 campos base, indice de paredes) UNA sola vez:

    fisica RK4      RK4_sim_basis_batch + electrode_geometry
    caracterizador  beam_progress_score + beam_characterization
    control         orchestrator (SIMION + Optuna) + starting_point
    datos           studies/ (estudios persistentes), outputs/ (resultados)

Uso tipico (desde la raiz del proyecto):

    from gemelo import GemeloDigital
    tw = GemeloDigital()

    tw.mejor()                  # mejor config del estudio persistente
    tw.predecir(voltajes)       # gemelo RK4: objetivo predicho + features
    tw.evaluar(voltajes)        # SIMION real: objetivo + hits + features ricas
    tw.entrenar(presupuesto=50) # corre el lazo completo (RK4 filtra,
                                #   SIMION confirma, Optuna aprende)
    tw.sembrar(voltajes, objetivo=..., hits=...)  # config CONOCIDO -> trial
    tw.sembrar_desde_registro() # vuelca todas las corridas SIMION del
                                #   registro al estudio (conocimiento previo)

`voltajes` puede ser un dict {electrodo: V} SOLO con los optimizables
(3, 6, 9, 10, 11, 12, 15, 18) o un array de 19; los electrodos fijos
(fuente +500, detector -2000, tierras) se imponen siempre.

Demo:
    python gemelo.py            # resumen + prediccion RK4 del mejor config
    python gemelo.py --evaluar  # ademas lo vuela en SIMION real
"""

import argparse
import datetime
import json
import pathlib
import sys
import optuna

import numpy as np

import optimizer as op
import orchestrator as orc
from RK4_sim_basis import IonSpecies
from RK4_sim_basis_batch import (
    BatchBasisFieldMap, make_batch_beam, BatchTrajectory, BatchRK4Integrator,
)
from beam_progress_score import make_beam, BeamProgressScorer
from beam_characterization import characterize_beam
from caracterizador import caracterizar, desde_simion_ultimo_vuelo, objetivo_v2
from electrode_geometry import build_wall_index

HERE = pathlib.Path(__file__).resolve().parent

# Registro persistente de TODA corrida hecha via la fachada (evaluar y
# predecir): una linea JSON por corrida con fecha, voltajes, objetivo y
# features. Es la memoria del gemelo fuera de los estudios Optuna -- y de
# paso, el dataset que la calibracion de pesos (Task B) consume gratis.
REGISTRO = HERE / "studies" / "registro_corridas.jsonl"


def _next_study_name(base_name: str, studies_dir: pathlib.Path):
    studies_dir.mkdir(exist_ok=True)
    base_db = studies_dir / f"{base_name}.db"
    if not base_db.exists():
        return base_name, base_db
    i = 1
    while True:
        candidate_name = f"{base_name}_{i}"
        candidate_db = studies_dir / f"{candidate_name}.db"
        if not candidate_db.exists():
            return candidate_name, candidate_db
        i += 1


def _find_best_existing_study_db(base_name: str, studies_dir: pathlib.Path):
    studies_dir.mkdir(exist_ok=True)
    import glob
    import optuna

    pattern = f"{base_name}*.db"
    db_paths = sorted(glob.glob(str(studies_dir / pattern)))
    best = None
    best_value = None
    best_direction = None
    best_name = None
    for db_path in db_paths:
        try:
            summaries = optuna.study.get_all_study_summaries(storage=f"sqlite:///{db_path}")
        except Exception:
            continue
        for summary in summaries:
            try:
                study = optuna.load_study(study_name=summary.study_name, storage=f"sqlite:///{db_path}")
            except Exception:
                continue
            if not study.trials:
                continue
            trial_values = [t.value for t in study.trials if t.value is not None]
            if not trial_values:
                continue
            direction = study.direction
            current_best = min(trial_values) if direction == optuna.study.StudyDirection.MINIMIZE else max(trial_values)
            if best is None:
                best = db_path
                best_name = summary.study_name
                best_value = current_best
                best_direction = direction
            else:
                if best_direction == optuna.study.StudyDirection.MINIMIZE:
                    if current_best < best_value:
                        best = db_path
                        best_name = summary.study_name
                        best_value = current_best
                        best_direction = direction
                else:
                    if current_best > best_value:
                        best = db_path
                        best_name = summary.study_name
                        best_value = current_best
                        best_direction = direction
    if best is None:
        return None
    return best_name, pathlib.Path(best)


class GemeloDigital:
    """Fachada del gemelo: fisica cargada una vez, estudio persistente."""

    def __init__(self, estudio=None, db=None):
        """estudio/db: nombre y sqlite del estudio Optuna. None = los del
        orchestrator (orchestrator_loop en studies/)."""
        if estudio is None and db is None:
            fallback = _find_best_existing_study_db(orc.STUDY_NAME, HERE / "studies")
            if fallback is not None:
                self.estudio, self.db = fallback
            else:
                self.estudio = orc.STUDY_NAME
                self.db = orc.RESULTS_DB
        else:
            self.estudio = estudio or orc.STUDY_NAME
            self.db = pathlib.Path(db) if db else orc.RESULTS_DB
        self._bfm = None
        self._wall = None
        self._margen = 1.5   # lo fija cargar_fisica() al cargar la fisica
        self._species = IonSpecies(mass=28 * 1.66053906660e-27,
                                   charge=1.602176634e-19)

    # ------------------------------------------------------------------
    # carga perezosa de la fisica (los 19 CSVs tardan ~40 s; una sola vez)
    # dual 1mm si basis_quad/ existe, clasica si no -- ver dual_grid
    # ------------------------------------------------------------------
    @property
    def fisica(self):
        if self._bfm is None:
            print("[gemelo] cargando fisica...")
            from dual_grid import cargar_fisica # type: ignore
            fis = cargar_fisica(HERE, n_electrodes=orc.N_ELECTRODES)
            self._bfm, self._wall, self._margen = fis.bfm, fis.wall, fis.margen
        return self._bfm, self._wall

    # ------------------------------------------------------------------
    def voltajes_completos(self, voltajes):
        """dict {electrodo:V} (solo optimizables) o array(19) -> array(19)
        con los FIJOS impuestos siempre."""
        v = np.zeros(orc.N_ELECTRODES)
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

    # ------------------------------------------------------------------
    def predecir(self, voltajes, particulas=200, pasos=None, seed=1234):
        """
        El gemelo RK4: vuela `voltajes` y devuelve la prediccion del
        objetivo real (mm, menor = mejor) + la caracterizacion del haz.
        No toca SIMION. ~10-15 s a 200 particulas (mas la carga inicial).
        """
        bfm, wall = self.fisica
        v = self.voltajes_completos(voltajes)
        pasos = pasos or orc.RESCREEN_STEPS
        dt = orc.RESCREEN_DT

        pos0, vel0 = make_beam(N=particulas, species=self._species,
                               start_point=[395.0, 75.0, 77.0], mean_energy_eV=15.0,
                               std_energy_eV=0.42466, half_angle_deg=15.0, seed=seed)
        bfm.set_voltages_batch(v[None, :])
        beam, ci = make_batch_beam(self._species, pos0, vel0, 1)
        tray = BatchTrajectory(beam, ci)
        BatchRK4Integrator(bfm, ci).integrate(tray, dt=dt, num_steps=pasos)

        scorer = BeamProgressScorer(
            bfm=bfm, Trajectory=tray, dt=dt, num_steps=pasos,
            detector_bbox=orc.DETECTOR_BBOX, wall_index=wall,
            wall_hit_margin=self._margen,
            wall_check_midpoints=False, wall_check_stride=3,
        )
        r = scorer.combined_score(v[None, :], **orc.SCORE_WEIGHTS)

        tdist = r["target_distance"]
        reached = float(r["reached_target"].mean())
        wall_frac = float(r["hit_wall"].mean())
        n_keep = max(1, int(np.ceil(particulas * orc.SPLAT_TOP_FRACTION)))

        # estado final por particula (posicion y velocidad al resolverse),
        # medido con EL MISMO caracterizador que usa el lado SIMION.
        # mascara = particulas "limpias" (sin choque ni perdida) cuando hay
        # suficientes: en el RK4 la particula no muere al chocar y sin el
        # filtro las features se contaminan con trayectorias muertas
        # (medido: sigma_x pasaba de ~3 a ~34 mm).
        posiciones = r["positions"]
        stop = r["stop_idx"]
        idx = np.arange(posiciones.shape[1])
        vels = np.array([s.velocity for s in tray.states])  # (T, N, 3) mm/s
        limpias = ~r["hit_wall"] & ~r["lost"]
        features = caracterizar(
            posiciones[stop, idx], vels[stop, idx], pos0, vel0,
            mascara=limpias if limpias.sum() >= 3 else None,
            flags=dict(reached=r["reached_target"], lost=r["lost"],
                       hit_wall=r["hit_wall"]),
        )
        features["n_limpias"] = int(limpias.sum())
        # acercamiento RK4 = maximo acercamiento de trayectoria (la
        # particula no muere, su posicion final no es un splat)
        features["dist_punta_mm"] = float(np.sort(tdist)[:n_keep].mean())
        objetivo, desglose = objetivo_v2(features)  # con pared (flags presentes)

        resultado = dict(fuente="rk4", objetivo=objetivo, desglose=desglose,
                         reach_fraction=reached, wall_fraction=wall_frac,
                         features=features,
                         voltajes={e: round(float(v[e - 1]), 1) for e in sorted(op.OPTIMIZE)})
        self._registrar(resultado)
        return resultado

    # ------------------------------------------------------------------
    def evaluar(self, voltajes):
        """
        SIMION real (~6 s): vuela `voltajes`, devuelve el objetivo real,
        hits, y la caracterizacion RICA (con velocidades del recording).
        """
        v = self.voltajes_completos(voltajes)
        chosen = {e: float(v[e - 1]) for e in op.OPTIMIZE}
        orc.apply_voltages(chosen)
        out = orc.run_simion(orc.FLY_COMMAND)
        posiciones = op.get_positions(out)
        print(f"[gemelo] SIMION: {posiciones.shape[0]} particulas resueltas")
        if posiciones.shape[0] == 0:
            return dict(fuente="simion", objetivo=orc.BAD_SCORE, hits=0, features=None)

        hits = int(op.count_hits(posiciones))
        try:
            # recording rico: estado inicial+final por ion -> caracterizacion
            # FULL con el caracterizador unico (incluye Twiss y residuos)
            _, features = desde_simion_ultimo_vuelo()
        except Exception:
            features = caracterizar(posiciones)  # degradado: sin velocidades
        objetivo, desglose = objetivo_v2(features, con_pared=False)
        resultado = dict(fuente="simion", objetivo=float(objetivo),
                         desglose=desglose, hits=hits,
                         mean_splat_all=orc.mean_splat_distance(posiciones),
                         features=features, voltajes=chosen)
        self._registrar(resultado)
        return resultado

    # ------------------------------------------------------------------
    # memoria de corridas: cada evaluar()/predecir() queda anotado solo
    # ------------------------------------------------------------------
    def _registrar(self, resultado):
        """Anota una corrida en studies/registro_corridas.jsonl (una linea
        JSON por corrida, con fecha). Nunca falla la corrida por el registro."""
        try:
            fila = dict(fecha=datetime.datetime.now().isoformat(timespec="seconds"),
                        estudio=self.estudio, **resultado)
            REGISTRO.parent.mkdir(exist_ok=True)
            with open(REGISTRO, "a", encoding="utf-8") as f:
                f.write(json.dumps(fila, default=float) + "\n")
        except Exception as exc:
            print(f"[gemelo] aviso: no se pudo registrar la corrida ({exc})")

    def historial(self, fuente=None, ultimos=None):
        """
        Relee las corridas registradas (lo que preguntabas: guardar una
        corrida y usarla despues). Devuelve una lista de dicts, la mas
        vieja primero.

        fuente : "rk4" | "simion" | None (todas)
        ultimos: int -- solo las N mas recientes
        """
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

    # ------------------------------------------------------------------
    # siembra: arrancar el estudio con conocimiento previo en vez de cero
    # ------------------------------------------------------------------
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
        """
        Agrega un config con resultado SIMION CONOCIDO como trial real del
        estudio, para que entrenar() arranque desde el en vez de desde
        cero. `objetivo` debe ser el J_v2 medido (la moneda del estudio).
        Duplicados (0.1V) se ignoran. Devuelve True si se agrego.
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction=orc.DIRECTION, storage=f"sqlite:///{self.db}",
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
        """
        Vuelca al estudio TODAS las corridas SIMION reales del registro
        (studies/registro_corridas.jsonl) que aun no esten en el: los
        buenos le dicen a Optuna donde esta el tesoro, los malos que
        evitar. Devuelve cuantas se sembraron.
        """
        sembradas = 0
        for fila in self.historial(fuente="simion"):
            volts = fila.get("voltajes")
            objetivo = fila.get("objetivo")
            if not volts or objetivo is None:
                continue
            if self.sembrar(volts, objetivo, hits=fila.get("hits"),
                            features=fila.get("features")):
                sembradas += 1
        print(f"[gemelo] {sembradas} corridas SIMION del registro sembradas "
              f"en el estudio '{self.estudio}'")
        return sembradas

    # ------------------------------------------------------------------
    def entrenar(self, presupuesto=20, por_iteracion=10, **kwargs):
        """Corre el lazo completo sobre el estudio de esta instancia y lo
        deja mas informado (todo queda persistido en la DB)."""
        orc.STUDY_NAME = self.estudio
        orc.RESULTS_DB = self.db
        return orc.orchestrate(total_simion_budget=presupuesto,
                               simion_per_iteration=por_iteracion, **kwargs)

    # ------------------------------------------------------------------
    def mejor(self):
        """Mejor trial del estudio persistente (o None si esta vacio)."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if not self.db.exists():
            return None
        try:
            study = optuna.load_study(study_name=self.estudio, storage=f"sqlite:///{self.db}")
        except KeyError:
            return None  # la DB existe pero no contiene este estudio
        done = [t for t in study.trials if t.value is not None]
        if not done:
            return None
        # respeta la direccion del estudio: los viejos maximizaban hits,
        # los actuales minimizan la distancia de splat (mm)
        if study.direction == optuna.study.StudyDirection.MAXIMIZE:
            best = max(done, key=lambda t: t.value)
        else:
            best = min(done, key=lambda t: t.value)
        return dict(trial=best.number, objetivo=best.value,
                    hits=best.user_attrs.get("simion_hits"),
                    voltajes={k: round(vv, 1) for k, vv in sorted(
                        best.params.items(), key=lambda kv: int(kv[0][1:]))},
                    n_trials=len(done))

    # ------------------------------------------------------------------
    def mejor_global(self):
        """Mejor trial POR HITS a traves de los estudios archivados del mismo nombre.
        Busca solo archivos de estudio bajo el nombre actual (p. ej. gemelo_v2*.db)."""
        import glob
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        mejor = None
        pattern = f"{self.estudio}*.db"
        dbs = sorted(glob.glob(str(HERE / "studies" / pattern))) + \
              sorted(glob.glob(str(HERE / "legacy" / "studies" / pattern)))
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
    parser = argparse.ArgumentParser(description="Gemelo digital CLI")
    parser.add_argument("--new-study", action="store_true",
                        help="Create a new Optuna study DB under studies/ instead of reusing the existing one.")
    parser.add_argument("--evaluar", action="store_true",
                        help="Run real SIMION evaluation for the selected best trial.")
    args = parser.parse_args()

    if args.new_study:
        study_name, db_path = _next_study_name(orc.STUDY_NAME, HERE / "studies")
        print(f"Creando nuevo estudio '{study_name}' en {db_path}")
        tw = GemeloDigital(estudio=study_name, db=db_path)
        print(f"Estudio '{tw.estudio}' vacio -- creando la PRIMERA VERSION del "
              f"modelo v2 (20 corridas SIMION, ~7 min)...")
        tw.entrenar(presupuesto=20, por_iteracion=10)
        m = tw.mejor()
        if m is None:
            print("El entrenamiento no registro trials -- revisar SIMION.")
            sys.exit(1)
    else:
        tw = GemeloDigital()
        m = tw.mejor()
        if m is None:
            print(f"Estudio '{tw.estudio}' vacio -- no hay trials. Use --new-study para crear un nuevo estudio.")
            sys.exit(1)

    print(f"Estudio '{tw.estudio}': {m['n_trials']} trials, mejor = trial {m['trial']} "
          f"(J_v2={m['objetivo']:.3f}, hits={m['hits']})")
    print(f"  voltajes: {m['voltajes']}")

    print("\nPrediccion del gemelo RK4 para ese config:")
    pred = tw.predecir(m["voltajes"])
    print(f"  J_v2 predicho: {pred['objetivo']:.3f}  "
          f"(reach {pred['reach_fraction']*100:.0f}%, wall {pred['wall_fraction']*100:.0f}%)")
    for k, val in pred["features"].items():
        if isinstance(val, (int, np.integer)) or (isinstance(val, float) and not np.isnan(val)):
            print(f"    {k:16s} = {val:.3f}" if isinstance(val, float) else f"    {k:16s} = {val}")

    if args.evaluar:
        print("\nEvaluacion SIMION real:")
        ev = tw.evaluar(m["voltajes"])
        print(f"  J_v2 real: {ev['objetivo']:.3f}  hits={ev['hits']}")
        for k, val in ev["features"].items():
            if isinstance(val, float) and not np.isnan(val):
                print(f"    {k:16s} = {val:.3f}")
