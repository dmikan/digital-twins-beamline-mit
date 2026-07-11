"""
gemelo_entrenable.py
=====================

Gemelo digital ENTRENABLE sobre los estudios persistidos en studies/.

Extiende la fachada GemeloDigital (gemelo.py, intacta) con la pieza que
faltaba: un modelo estadistico (GP de numpy, sin dependencias nuevas)
entrenado directamente sobre los trials del .db, que predice al instante
(sin volar RK4 ni SIMION) y declara su incertidumbre. Con esto el gemelo
queda completo en sus cuatro verbos:

  - interactuar con el DB : datos(), resumen(), sembrar() (en el espacio
                            de parametros PROPIO del estudio, p.ej. A/B/C
                            del bender7d -- la version heredada solo
                            escribia V3..V18), mejor() (heredado).
  - evaluar               : evaluar() SIMION real (heredado) y
                            evaluar_y_aprender(), que ademas siembra el
                            resultado en el DB y re-entrena -- y reporta
                            cuanto le erro el modelo a ese punto.
  - predecir              : predecir() fisico RK4 (heredado, ~segundos) y
                            predecir_modelo() estadistico (instantaneo,
                            con sigma) -- dos niveles de fidelidad.
  - entrenar              : entrenar_modelo() ajusta el GP contra el
                            historial del estudio con seleccion de
                            hiperparametros por leave-one-out exacto, y
                            entrenar() (heredado) sigue corriendo el loop
                            RK4+SIMION completo del orquestador.

Ademas: sugerir() propone candidatos nuevos por Expected Improvement del
GP, y ciclo() cierra el lazo sugerir -> evaluar en SIMION -> sembrar ->
re-entrenar.

Uso rapido (estudio central por defecto: bender7d / Search v2.1):

    from gemelo_entrenable import GemeloEntrenable
    tw = GemeloEntrenable()
    print(tw.resumen())
    print(tw.entrenar_modelo())              # ajusta el GP (objetivo J)
    print(tw.predecir_modelo(tw.mejor()["voltajes"]))
    for c in tw.sugerir(3): print(c)
    # tw.ciclo(rondas=1, k=2)                # consume 2 corridas SIMION

Demo sin SIMION:  python gemelo_entrenable.py
"""

import pathlib

import numpy as np
import optuna

import optimizer as op
from gemelo import GemeloDigital

HERE = pathlib.Path(__file__).resolve().parent

# Orden canonico de parametros (se filtra a los que el estudio realmente usa)
_ORDEN = ["V3", "V6", "A", "B", "C", "V9", "V10", "V11", "V12", "V15", "V18"]


# ----------------------------------------------------------------------
# GP minimo en numpy (RBF + ruido), con leave-one-out exacto
# ----------------------------------------------------------------------
class _GP:
    def __init__(self, ls, ruido):
        self.ls, self.ruido = ls, ruido

    def ajustar(self, X, y):
        self.X = X
        self.y_mu, self.y_sd = float(y.mean()), float(y.std() or 1.0)
        self.y = (y - self.y_mu) / self.y_sd
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        K = np.exp(-0.5 * d2 / self.ls ** 2) + (self.ruido ** 2) * np.eye(len(X))
        self.Ki = np.linalg.inv(K)
        self.alpha = self.Ki @ self.y
        return self

    def predecir(self, Xq):
        d2 = ((Xq[:, None, :] - self.X[None, :, :]) ** 2).sum(-1)
        Ks = np.exp(-0.5 * d2 / self.ls ** 2)
        mu = Ks @ self.alpha
        var = 1.0 + self.ruido ** 2 - np.einsum("ij,jk,ik->i", Ks, self.Ki, Ks)
        sd = np.sqrt(np.maximum(var, 1e-9))
        return mu * self.y_sd + self.y_mu, sd * self.y_sd

    def loo(self):
        """Leave-one-out exacto (Rasmussen & Williams eq. 5.10-5.12)."""
        kii = np.diag(self.Ki)
        mu = self.y - self.alpha / kii
        return mu * self.y_sd + self.y_mu


class GemeloEntrenable(GemeloDigital):
    """GemeloDigital + modelo estadistico entrenable sobre el .db."""

    def __init__(self, estudio="gemelo_v2_bender7d", db=None):
        db = db or (HERE / "studies" / f"{estudio}.db")
        super().__init__(estudio=estudio, db=db)
        self._modelo = {}     # {"J": _GP, "hits": _GP}
        self._nombres = None  # nombres de parametros del estudio
        self._cotas = None    # (lo, hi) por parametro

    # ------------------------------------------------------------------
    # interactuar con el DB
    # ------------------------------------------------------------------
    def _study(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        return optuna.load_study(study_name=self.estudio,
                                 storage=f"sqlite:///{self.db}")

    def datos(self):
        """Trials COMPLETE del estudio como matrices para entrenar.

        Returns dict: X (n,d), y (J), hits (n, con NaN si no registrado),
        nombres, cotas (lo, hi), trials (números).
        """
        s = self._study()
        done = [t for t in s.trials
                if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        if not done:
            raise RuntimeError(f"estudio '{self.estudio}' sin trials completos")
        nombres = [p for p in _ORDEN if p in done[0].params]
        filas, y, hits, nums = [], [], [], []
        for t in done:
            if not all(n in t.params for n in nombres):
                continue
            filas.append([float(t.params[n]) for n in nombres])
            y.append(float(t.value))
            h = t.user_attrs.get("simion_hits")
            hits.append(float(h) if h is not None else np.nan)
            nums.append(t.number)
        d0 = done[0].distributions
        lo = np.array([d0[n].low for n in nombres])
        hi = np.array([d0[n].high for n in nombres])
        self._nombres, self._cotas = nombres, (lo, hi)
        return dict(X=np.array(filas), y=np.array(y), hits=np.array(hits),
                    nombres=nombres, cotas=(lo, hi), trials=nums)

    def resumen(self):
        d = self.datos()
        con_h = d["hits"][~np.isnan(d["hits"])]
        return dict(estudio=self.estudio, db=str(self.db), n_trials=len(d["y"]),
                    parametros=d["nombres"], mejor_J=float(d["y"].min()),
                    mejor_hits=int(con_h.max()) if len(con_h) else None,
                    modelo_entrenado=sorted(self._modelo))

    def sembrar(self, voltajes, objetivo, hits=None, features=None):
        """Siembra un resultado como trial nuevo EN EL ESPACIO DE PARAMETROS
        DEL ESTUDIO (la version heredada escribia siempre V3..V18 en 8D --
        inservible para los estudios reducidos A/B/C como el central)."""
        if self._nombres is None:
            self.datos()
        v = self.voltajes_completos(voltajes)
        if any(n in ("A", "B", "C") for n in self._nombres):
            red = op._full_to_reduced_row(v)
            plano = {f"V{k}" if isinstance(k, int) else k: float(x)
                     for k, x in red.items()}
        else:
            plano = {f"V{e}": float(v[e - 1]) for e in op.OPTIMIZE}
        params = {n: plano[n] for n in self._nombres}

        s = self._study()
        clave = tuple(round(params[n], 1) for n in self._nombres)
        for t in s.get_trials(deepcopy=False):
            if all(n in t.params for n in self._nombres) and \
               tuple(round(t.params[n], 1) for n in self._nombres) == clave:
                return False
        ref = next(t for t in s.trials
                   if t.state == optuna.trial.TrialState.COMPLETE)
        attrs = {"sembrado": True}
        if hits is not None:
            attrs["simion_hits"] = int(hits)
        for fk in self._FEATURES_SEMILLA:
            val = (features or {}).get(fk)
            if val is not None and np.isfinite(val):
                attrs[f"f_{fk}"] = float(val)
        s.add_trial(optuna.trial.create_trial(
            params=params,
            distributions={n: ref.distributions[n] for n in self._nombres},
            value=float(objetivo), user_attrs=attrs))
        return True

    # ------------------------------------------------------------------
    # entrenar / predecir (modelo estadistico)
    # ------------------------------------------------------------------
    def _normalizar(self, X):
        lo, hi = self._cotas
        return 2.0 * (X - lo) / (hi - lo) - 1.0

    def entrenar_modelo(self, objetivo="J",
                        grilla_ls=(0.15, 0.3, 0.6, 1.2),
                        grilla_ruido=(0.05, 0.15, 0.3)):
        """Ajusta el GP sobre el historial del estudio. Seleccion de
        (length_scale, ruido) por RMSE leave-one-out EXACTO. Devuelve las
        metricas de entrenamiento -- el Spearman LOO es la honestidad del
        modelo: con cuanta fidelidad ordena puntos que no vio."""
        from scipy.stats import spearmanr
        d = self.datos()
        if objetivo == "hits":
            mask = ~np.isnan(d["hits"])
            X, y = d["X"][mask], d["hits"][mask]
        else:
            X, y = d["X"], d["y"]
        Xn = self._normalizar(X)
        mejor = None
        for ls in grilla_ls:
            for rz in grilla_ruido:
                gp = _GP(ls, rz).ajustar(Xn, y)
                rmse = float(np.sqrt(((gp.loo() - y) ** 2).mean()))
                if mejor is None or rmse < mejor[0]:
                    mejor = (rmse, ls, rz, gp)
        rmse, ls, rz, gp = mejor
        self._modelo[objetivo] = gp
        rho = float(spearmanr(gp.loo(), y).statistic)
        return dict(objetivo=objetivo, n=len(y), length_scale=ls, ruido=rz,
                    rmse_loo=round(rmse, 4), spearman_loo=round(rho, 3))

    def _vector_params(self, voltajes):
        if self._nombres is None:
            self.datos()
        if isinstance(voltajes, dict) and all(n in voltajes for n in self._nombres):
            return np.array([float(voltajes[n]) for n in self._nombres])
        v = self.voltajes_completos(voltajes)
        if any(n in ("A", "B", "C") for n in self._nombres):
            red = op._full_to_reduced_row(v)
            plano = {f"V{k}" if isinstance(k, int) else k: float(x)
                     for k, x in red.items()}
        else:
            plano = {f"V{e}": float(v[e - 1]) for e in op.OPTIMIZE}
        return np.array([plano[n] for n in self._nombres])

    def predecir_modelo(self, voltajes, objetivo="J"):
        """Prediccion INSTANTANEA del modelo estadistico, con incertidumbre.
        (Para la prediccion fisica RK4, usar predecir() -- heredada.)"""
        if objetivo not in self._modelo:
            self.entrenar_modelo(objetivo=objetivo)
        x = self._vector_params(voltajes)
        mu, sd = self._modelo[objetivo].predecir(self._normalizar(x[None, :]))
        return {f"{objetivo}_pred": float(mu[0]), "sigma": float(sd[0])}

    # ------------------------------------------------------------------
    # sugerir / evaluar-y-aprender / ciclo
    # ------------------------------------------------------------------
    def sugerir(self, n=5, pool=800, sigma_frac=0.10, semilla=None):
        """Candidatos nuevos por Expected Improvement del GP (minimizar J):
        pool de perturbaciones alrededor del mejor (70%) + uniformes (30%)."""
        d = self.datos()
        if "J" not in self._modelo:
            self.entrenar_modelo("J")
        gp = self._modelo["J"]
        lo, hi = self._cotas
        rng = np.random.default_rng(semilla)
        mejor_x = d["X"][int(np.argmin(d["y"]))]

        n_loc = int(pool * 0.7)
        Xc = np.vstack([
            mejor_x + rng.normal(0, sigma_frac * (hi - lo), size=(n_loc, len(lo))),
            rng.uniform(lo, hi, size=(pool - n_loc, len(lo))),
        ])
        Xc = np.clip(Xc, lo, hi)
        mu, sd = gp.predecir(self._normalizar(Xc))

        from scipy.stats import norm
        y_best = float(d["y"].min())
        z = (y_best - mu) / sd
        ei = (y_best - mu) * norm.cdf(z) + sd * norm.pdf(z)
        top = np.argsort(ei)[::-1][:n]
        return [dict(params={k: round(float(v), 1) for k, v in
                             zip(d["nombres"], Xc[i])},
                     J_pred=round(float(mu[i]), 4), sigma=round(float(sd[i]), 4),
                     ei=round(float(ei[i]), 5)) for i in top]

    def evaluar_y_aprender(self, voltajes, reentrenar=True):
        """SIMION real -> siembra el resultado en el DB -> re-entrena.
        Reporta tambien el error de la prediccion previa del modelo."""
        pred = self.predecir_modelo(voltajes)
        res = self.evaluar(voltajes)
        sembrado = self.sembrar(voltajes, res["objetivo"], hits=res.get("hits"),
                                features=res.get("features"))
        info = dict(J_predicho=pred["J_pred"], sigma=pred["sigma"],
                    J_real=res["objetivo"], hits=res.get("hits"),
                    error=round(res["objetivo"] - pred["J_pred"], 4),
                    sembrado=sembrado)
        if reentrenar and sembrado:
            info["reentrenado"] = self.entrenar_modelo("J")
        return info

    def ciclo(self, rondas=1, k=3, **kw_sugerir):
        """Lazo completo: sugerir -> evaluar en SIMION el top-k -> sembrar ->
        re-entrenar. Consume rondas*k corridas reales de SIMION."""
        historial = []
        for r in range(1, rondas + 1):
            print(f"[ciclo] ronda {r}/{rondas}")
            for cand in self.sugerir(n=k, **kw_sugerir):
                info = self.evaluar_y_aprender(cand["params"])
                print(f"  J_pred={info['J_predicho']:.3f}(+-{info['sigma']:.3f})"
                      f" -> J_real={info['J_real']:.3f} hits={info['hits']}")
                historial.append(info)
        return historial


if __name__ == "__main__":
    tw = GemeloEntrenable()
    print("resumen:", tw.resumen())
    print("\nentrenamiento del modelo J:", tw.entrenar_modelo("J"))
    print("entrenamiento del modelo hits:", tw.entrenar_modelo("hits"))

    m = tw.mejor()
    print(f"\nmejor trial del estudio: #{m['trial']} J={m['objetivo']:.4f} hits={m['hits']}")
    print("prediccion del modelo para ese punto:",
          tw.predecir_modelo(m["voltajes"]),
          tw.predecir_modelo(m["voltajes"], objetivo="hits"))

    print("\nsugerencias por Expected Improvement:")
    for c in tw.sugerir(3, semilla=0):
        print(" ", c)
    print("\n(para cerrar el lazo con SIMION real: tw.ciclo(rondas=1, k=2))")
