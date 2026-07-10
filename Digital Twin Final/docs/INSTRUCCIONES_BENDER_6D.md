# Plan: reparametrizar el bender a 6D (offset + flexión) en optimizer.py

Fecha: 2026-07-08. Si esta sesion se corta, seguir esta lista en orden;
cada paso dice donde va el codigo y como verificarlo.

## Contexto / por que

Hoy el sampler (TPE o GP) busca en 8 dimensiones libres:
`V3, V6, V9, V10, V11, V12, V15, V18`.

Los 4 electrodos del bender (9,10,11,12) no son independientes: PCA sobre
130 configs reales con `simion_hits>0` (de `studies/gemelo_v2.db` +
`studies/gemelo_db_v2.db`) muestra que 2 direcciones explican 98.5% de la
varianza, y la mejor pareja de modos **puramente analiticos** (geometria +
una sola medicion de campo, cero historial de trials) es:

- **monopolo** `(+1,+1,+1,+1)` (offset comun del bender) -> 71.3% solo
- **quad/diagonal** `(+1,-1,-1,+1)` (patron de pares 9≡12, 10≡11 opuestos,
  ya derivado en `tools/bender_field_analysis.py` a partir de que par
  empuja el haz hacia +z) -> mejor complemento, 81.6% combinado

Mapa lineal fijo (sin termino base, A ya cubre cualquier offset):
```
V9  = A + B
V10 = A - B
V11 = A - B
V12 = A + B
```
donde `A` = offset comun, `B` = intensidad de flexion.

Esto reduce la busqueda de 8D a **6D**: `{V3, V6, A, B, V15, V18}`.

Evidencia/derivacion completa esta en la conversacion de esta sesion
(scripts temporales en la carpeta scratchpad, no en el repo). Si hace
falta reproducir el PCA o el chequeo de varianza explicada, recrear:
1. leer todos los trials COMPLETE de `studies/gemelo_v2.db` y
   `studies/gemelo_db_v2.db` con `optuna.study.get_all_study_summaries` +
   `optuna.load_study` (igual que hace `tools/report_figures.py`)
2. filtrar por `user_attrs["simion_hits"] > 0`
3. armar matriz `(V9,V10,V11,V12)`, centrar, SVD -> varianza explicada
4. proyectar sobre monopolo/dipolo_x/dipolo_z/quad (vectores +-1
   normalizados) para chequear que `monopole+quad` es la mejor pareja.

## Decision de diseno (importante, no cambiar sin discutirlo)

**Solo `optimizer.py` cambia.** `physics.py`, `caracterizador.py` y
`gemelo.py` NO se tocan: siguen operando sobre el vector fisico completo
de 19 voltajes (`V9..V12` reales), igual que hoy. La reparametrizacion
A/B es un detalle interno de la capa de busqueda de `optimizer.py`
(que arma los candidatos que se le mandan a RK4/SIMION), no del motor de
fisica ni del facade.

**Estudio/DB nuevo, no tocar `gemelo_v2`/`gemelo_db_v2`.** El espacio de
parametros de Optuna cambia de nombre (`V9..V12` -> `A, B`), asi que es un
espacio de busqueda DISTINTO. Mezclarlo en el mismo study rompe TPE/GP
(esperan params consistentes entre trials). Usar un study/db nuevo, p.ej.
`studies/gemelo_v2_bender6d.db`, study_name `gemelo_v2_bender6d`. Los
estudios viejos quedan intactos como estan.

## Pasos de implementacion en optimizer.py

1. **Constantes nuevas** (cerca de `OPTIMIZE`/`FIXED`, lineas ~49-66):
   - `BENDER_PLUS = (9, 12)`, `BENDER_MINUS = (10, 11)` (el patron quad)
   - `OPTIMIZE_ELECTRODES = (3, 6, 9, 10, 11, 12, 15, 18)` — lista fisica
     de electrodos libres, SEPARADA del dict de busqueda. La usan
     `run_simion_candidate`/`apply_voltages` (SIMION siempre necesita
     voltajes fisicos, sin importar la parametrizacion de busqueda).
   - `REDUCED_OPTIMIZE = {3: (-1000,1000), 6: (-1000,1000),
     "A": (-1000,1000), "B": (-1000,1000), 15: (-1000,1000),
     18: (-1000,1000)}` — el nuevo espacio de busqueda (6 keys, mixed
     int/str). OJO: los rangos de A y B no son necesariamente +-1000;
     revisar que el mapa V=A+-B no se salga de +-1000 en V9..V12 mas de
     lo razonable (con A,B en +-1000 el peor caso es V=+-2000, fuera del
     rango fisico original) — considerar A,B en +-500 cada uno, o clipear
     V9..V12 resultantes a +-1000 despues de expandir.

2. **Funciones de expansion/colapso** (nuevas, cerca de `load_starting_point`):
   ```python
   def expand_bender(A, B):
       return {9: A + B, 10: A - B, 11: A - B, 12: A + B}

   def collapse_bender(v9, v10, v11, v12):
       # proyeccion L2 sobre el plano monopolo+quad (exacta si el punto
       # ya esta en el plano; la mejor aproximacion si no lo esta)
       A = (v9 + v10 + v11 + v12) / 4.0
       B = (v9 - v10 - v11 + v12) / 4.0
       return A, B

   def expand_reduced_to_full(reduced: dict) -> np.ndarray:
       # reduced tiene keys 3, 6, "A", "B", 15, 18
       v = np.zeros(19)
       for k, val in FIXED.items():
           v[k - 1] = val
       v[3 - 1] = reduced[3]
       v[6 - 1] = reduced[6]
       bender = expand_bender(reduced["A"], reduced["B"])
       for e, val in bender.items():
           v[e - 1] = val
       v[15 - 1] = reduced[15]
       v[18 - 1] = reduced[18]
       return v
   ```

3. **`sample_voltage_batch`** (linea ~302): reemplazar el loop sobre
   `OPTIMIZE` por uno sobre `REDUCED_OPTIMIZE`, sampleando cada key
   (incluyendo `"A"`/`"B"` como param names literales `"A"`, `"B"` en
   `sampler.sample_independent(..., param_name=key if isinstance(key,str)
   else f"V{key}", ...)`), acumular en un dict `reduced_row`, expandir con
   `expand_reduced_to_full` a la fila de `batch`. Devolver TAMBIEN el
   array/dict de filas reducidas (paralelo a `voltages_batch`) — hace
   falta mas abajo para loguear los trials no-GP.

4. **`sample_voltage_batch_gp_seeded`** (linea ~316): mismo cambio.
   `trial.suggest_float("A", low, high)` / `trial.suggest_float("B", ...)`
   en vez de `V9..V12`. Guardar el dict reducido por fila (paralelo a
   `voltages_batch`) para las filas de perturbacion tambien (la
   perturbacion gaussiana debe aplicarse en A,B, no en V9..V12 directo).

5. **`inject_starting_point`** (linea ~196): hoy perturba directo en
   `OPTIMIZE` (space fisico). Cambiar a version reducida: convertir
   `sp`/`best_center` (vectores fisicos de 19, vienen de
   `derived_starting_point.json` o del mejor trial historico) a `(A,B)`
   con `collapse_bender`, perturbar A,B con ruido gaussiano, clipear en
   los rangos de `REDUCED_OPTIMIZE`, expandir de nuevo con
   `expand_reduced_to_full` antes de escribir en `voltages_batch`.

6. **`orchestrate()`** (linea ~416+):
   - `gp_seeded = isinstance(...)`: bug preexistente, no relacionado,
     dejarlo (ya esta flageado en memoria, no forma parte de este cambio).
   - Reconstruccion de `best_center` (linea ~465-474): hoy lee
     `trial.params[f"V{e_num}"]` para `e_num in OPTIMIZE`. Cambiar a leer
     `trial.params["A"]`, `trial.params["B"]`, `trial.params["V3"]`, etc.
     y armar el vector fisico completo con `expand_reduced_to_full`.
   - Set `seen` para dedup (linea ~502-506) y el key de duplicados
     (linea ~511): cambiar de `tuple(... for e in OPTIMIZE)` a la tupla
     reducida `(V3, V6, A, B, V15, V18)` redondeada, comparando
     `t.params` (que ahora tiene A,B) contra el `reduced_row` de la fila
     candidata (NO contra `voltages_batch[idx][8,9,10,11]`, que ya no es
     el espacio de busqueda).
   - Creacion de trial para promovidos NO-seed (linea ~569-570): cambiar
     `params = {f"V{e}": ... for e in OPTIMIZE}` a
     `params = {"V3": ..., "V6": ..., "A": reduced_row["A"], "B":
     reduced_row["B"], "V15": ..., "V18": ...}` — usar el valor reducido
     que efectivamente genero esa fila, no volver a colapsar desde el
     vector fisico (para que quede exactamente lo que el sampler "vio").
   - `run_simion_candidate` (linea ~283): sigue recibiendo el vector
     fisico completo de 19 volt (`voltages`), sin cambios — construye
     `chosen` con `OPTIMIZE_ELECTRODES` (la lista fisica nueva) en vez de
     `OPTIMIZE`.

7. **Nuevo runner/estudio**: agregar un `if __name__ == "__main__":` o un
   script chico (`tools/run_bender6d.py`) que llame `orchestrate(...,
   studyname="gemelo_v2_bender6d", db_path=ROOT/"studies"/
   "gemelo_v2_bender6d.db")` para no pisar los estudios existentes.

## Verificacion antes de gastar SIMION real

- Sanity check offline: expandir con `expand_bender` un `A,B` de prueba y
  confirmar que si `A=B=0` da `V9=V10=V11=V12=0`; si `B` grande y `A=0`
  reproduce aproximadamente el patron original `s * pattern` de
  `bender_field_analysis.py` (mismo signo).
  `derived_starting_point.json` a `(A,B)` y de vuelta, confirmar
  reconstruccion exacta salvo redondeo.
- Caveat a documentar (no ocultarlo en el informe): el mejor config real
  conocido (133 hits: V9=117, V10=92, V11=145, V12=-606) NO cae
  exactamente sobre el plano monopolo+quad (81.6% de varianza, no 100%),
  asi que la busqueda reducida en 6D probablemente NO puede reproducir
  ese punto exacto — es un tradeoff consciente (buscador mas eficiente
  en general, a costa de no poder alcanzar configs fuera del plano).
  Vale la pena, despues de tener algunas corridas, comparar el mejor
  resultado de `gemelo_v2_bender6d` contra el 8D historico para medir si
  el tradeoff valio la pena (disciplina de "medir, no asumir" de siempre
  en este proyecto).

## Estado al momento de escribir esto

**IMPLEMENTADO Y VALIDADO OFFLINE (2026-07-08), sin gastar SIMION real
todavia.** Cambios hechos en `optimizer.py`:

- Constantes: `OPTIMIZE_ELECTRODES`, `BENDER_PLUS`/`BENDER_MINUS`,
  `REDUCED_OPTIMIZE` (A,B en +-500V cada uno -- asi V9..V12=A+-B nunca
  se pasa de +-1000V, el mismo rango fisico original).
- Funciones: `expand_bender`, `collapse_bender`, `expand_reduced_to_full`,
  `_full_to_reduced_row`, `_param_name`.
- Funciones de sampling reducidas (paralelas a las 8D, que quedan
  intactas): `sample_voltage_batch_reduced`, `sample_voltage_batch_gp_seeded_reduced`,
  `inject_starting_point_reduced`.
- `run_simion_candidate` ahora usa `OPTIMIZE_ELECTRODES` (lista fisica) en
  vez de `OPTIMIZE` (dict de busqueda) para construir `chosen` -- asi
  sigue funcionando igual sin importar el espacio de busqueda.
- `orchestrate()` tiene un parametro nuevo `reduced_search=False` que
  bifurca en cada uno de los puntos listados arriba (sampling,
  best_center, inject, dedup, creacion de trial). Con
  `reduced_search=False` el comportamiento es BIT A BIT el mismo que
  antes (nada cambio para gemelo_v2/gemelo_db_v2).
- Runner nuevo: `tools/run_bender6d.py` (study `gemelo_v2_bender6d`, db
  `studies/gemelo_v2_bender6d.db`, `total_simion_budget=50`,
  `simion_per_iteration=10` -- ajustar si se quiere otro presupuesto).

**Verificado offline (sin SIMION):**
- round-trip `expand_bender`/`collapse_bender` exacto en 4 casos de
  prueba.
- consistencia `voltages_batch` <-> `reduced_batch` tras sampling GP-seeded,
  perturbacion (`inject_starting_point_reduced`), y llaves de dedup.
- `rk4_score_all` corre sobre el batch expandido con la fisica real
  (CampoFino/ParedesPA) y da scores finitos.
- creacion de `optuna.trial.create_trial` con params `{V3,V6,A,B,V15,V18}`
  funciona y los guarda correctamente en un study de prueba.

**Caveat medido, no solo teorico:** el mejor config real conocido (133
hits: V9=117, V10=92, V11=145, V12=-606) tiene un residuo de **512.6 V**
respecto al plano monopolo+quad -- la busqueda reducida en 6D
probablemente NO puede reproducir ese punto exacto. El
`derived_starting_point.json` (patron fisico puro) SI cae casi sobre el
plano (A=-10.1, B=-258.0), como se esperaba por construccion.

**Pendiente (no hecho todavia, requiere decision del usuario):**
correr `tools/run_bender6d.py` de verdad (gasta presupuesto SIMION real).
No se corrio en esta sesion -- confirmar con el usuario antes de lanzarlo,
y decidir el presupuesto (`total_simion_budget`/`simion_per_iteration`).
Despues de correrlo, comparar el mejor resultado contra el 8D historico
(gemelo_v2/gemelo_db_v2) para medir si el tradeoff de dimensionalidad
valio la pena -- no asumirlo.
