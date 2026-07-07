# Instrucciones para la próxima sesión (2026-07-07)

Dos cambios pedidos, ambos en `optimizer.py` (que absorbió el orchestrator).
Verificados/medidos esta sesión. **No tocar mientras haya una corrida activa.**

---

## CAMBIO A — Promover 10 candidatos + los 5 seeds del GP (15 SIMION/iter)

**Motivo:** hoy los seeds que genera el GP (`study.ask()`) pasan por el filtro
RK4 y solo se evalúan en SIMION si quedan en el top-K; los demás se **podan
(PRUNED)** y el GP nunca recibe feedback de su propia propuesta. Como (1) el
RK4 es poco fiable en la región informativa (Spearman +0.18) y (2) el loop se
**atasca en el basin del punto de arranque** (medido: el best de 133 y el de 93
están a 1241 V, en basins de signo opuesto), evaluar los seeds del GP además
del top-K restaura el lazo Bayesiano y le da al GP el mecanismo para **saltar
de basin**.

**Config:** `simion_per_iteration = 10` (K=10 del RK4) y `n_gp_seeds = 5`
→ **15 corridas SIMION por iteración**.

**Dónde:** en `orchestrate()`, en el bloque de promoción, justo después de
calcular `top_k_idx` (el top-10 del RK4) y ANTES del loop
`for rank, idx in enumerate(top_k_idx, 1)`.

**Qué hacer:**
1. Unir el top-K del RK4 con los seeds del GP no promovidos, y evaluar TODOS
   en SIMION:
```python
# top_k_idx ya tiene los 10 del RK4. Agregar los seeds del GP no incluidos:
seeds_extra = [idx for idx in seed_trial_for_row if int(idx) not in set(int(i) for i in top_k_idx)]
top_k_idx = np.array(list(top_k_idx) + seeds_extra, dtype=int)   # 10 + (hasta 5) = 15
```
2. El loop de evaluación SIMION (`for rank, idx in enumerate(top_k_idx, 1)`)
   ya vuela todo lo que esté en `top_k_idx`, así que ahora vuela los 15. No
   cambia.
3. **Eliminar** (o dejar vacío) el bloque que podaba los seeds no promovidos —
   ya no hay seeds sin evaluar:
```python
# BORRAR esto (ya no aplica, todos los seeds se evaluan):
# for idx, trial in seed_trial_for_row.items():
#     if idx not in top_k_idx:
#         study.tell(trial, state=optuna.trial.TrialState.PRUNED)
```
4. **Ojo con el presupuesto:** ahora se gastan 15 SIMION/iter, no 10. Si
   `total_simion_budget` es fijo, habrá menos iteraciones. Ajustar el budget
   para el nº de iteraciones deseado (p.ej. budget = 15 × n_iter).

**Verificación:** una iteración debe imprimir "Promoting top 15..." y registrar
15 trials con `simion_hits`; ninguno con estado PRUNED por screening.

---

## CAMBIO B — Parámetros de velocidad óptima (calibrados a estas corridas)

Medido esta sesión con la física 2.5mm + fix de campo redundante:

| medición | valor |
|---|---|
| Stage-A screening (50p × 1500 pasos, chunk 32) | **360 ms/config** |
| Rescreen (200p × 3000 pasos, chunk 8) | **2755 ms/config** |
| SIMION por candidato | **~5.8 s** |
| costo marginal RK4 medido | **4.7e-6 s/(partícula·paso)** (raw) |

**Ajustes en `optimizer.py`:**

1. `RK4_MARGINAL_COST_PER_PARTICLE_STEP_S = 4.1e-6`  (era 2.8e-6)
   El 2.5mm cuesta ~1.5× más por partícula-paso que la física vieja (ruteo de
   cajas finas + wall check). Con 4.1e-6 × overhead 1.14, el modelo predice
   0.351 s/config stage-A y 2.80 s/config rescreen — coincide con lo medido.
   **Efecto:** el cost-planner elegirá una M (candidatos a screenear) más
   chica y realista, evitando sobre-presupuestar el screening.

2. `SIMION_COST_PER_CANDIDATE_S = 5.8`  (era 5.77, ok — dejar o redondear)

3. `CHUNK_FINO = 32` en `physics.py` — **ya está**, es el óptimo para 2.5mm
   (E_batch ~pequeño, throughput 2.88 → 1.78 s/config tras el fix). No bajar.

4. **El rescreen es ahora el costo dominante del screening** (2.75 s/config ×
   30 = ~82 s/iter, comparable al stage-A). Si se necesita más velocidad,
   bajar `RESCREEN_PARTICLES` de 200 a ~120 o `RESCREEN_STEPS` de 3000 a ~2500
   — pero validar antes que el ranking del rescreen no se degrade
   (playpen/validar_rk4… o comparar top-K promovidos).

**Balance de tiempo por iteración (con 15 SIMION):**
- SIMION: 15 × 5.8 = ~87 s
- El cost-planner apunta a rk4_time ≈ simion_time → con 0.36 s/config, M ≈ 240
  candidatos de stage-A (~87 s), + rescreen top-~45 (~124 s). Iteración total
  ~5 min. Si es mucho, reducir M vía el budget de tiempo o el rescreen.

---

## Contexto que NO hay que re-descubrir
- La física 2.5mm + paredes-del-PA (sin STL) ya está promovida en `physics.py`.
  Las bases 1mm están en `basis_quad/backup_1mm/` (reversible).
- El objetivo J_v2 v2.2 (transmisión empinada, H0_HITS=15) ya está corregido:
  Spearman(hits,J) = −0.87. NO volver a la versión lineal.
- Los .db fluctúan por OneDrive: para números de reporte, **congelar copias**.
- El twin es fiable para transmisión (dirección) pero NO para centroide
  (~7 mm de piso de error) — ver `docs/` y playpen/rango_confiable.py.
