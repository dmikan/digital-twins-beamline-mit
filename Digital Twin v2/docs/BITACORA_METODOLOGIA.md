# Bitácora de metodología — en palabras de Julián

Registro de decisiones para el informe final (deadline: 10 de julio).
Regla de trabajo: cada vez que hagamos un cambio o tomemos una decisión,
Julián escribe acá **en sus propias palabras y en español** qué estamos
haciendo y por qué. Claude pregunta, Julián redacta, Claude solo pega la
entrada tal cual (sin cambiar el lenguaje) y agrega los datos duros de
soporte debajo, marcados como [datos].

Formato de cada entrada:

```
## AAAA-MM-DD — <título corto>
<texto de Julián, sin editar>

[datos] <números/mediciones que respaldan la entrada, agregados por Claude>
```

---

## Entradas pendientes de redactar (backfill opcional)
Momentos ya ocurridos que valdría la pena tener en tus palabras, si querés
completarlos en algún descanso:

- [ ] Por qué congelamos el proyecto para el primer deadline y qué
      priorizamos entregar (guía de código + figuras + secciones 6/7).
- [ ] La decisión del umbral de 100 hits para estrechar la búsqueda
      (explorar amplio vs explotar fino).
- [ ] El cambio del objetivo de "contar hits" a "distancia media de splat"
      y luego al 10% más cercano + término de transmisión.

---

<!-- Las entradas nuevas van debajo de esta línea, la más reciente al final. -->

## 2026-07-04 — Ampliar la información extraída de cada corrida de SIMION

En este momento de la investigación se está ampliando la información que se
extrae de cada simulación de SIMION. Antes solo recuperamos hits, métrica
que como ya se observó es muy mala para predecir un beam que actúe como
queremos, por eso se va a guardar la información de la velocidad de SIMION
para construir el offset del centroide pues este guía la dirección del haz
en el espacio, califica bien aquellos cuyo cambio se da en sentido del
objetivo. Ahora, en este momento se quiere implementar una función
analizador de beam que sea la misma entre RK4 y SIMION, para que las
mediciones del target sean comparables entre simulaciones y que prediga
los hiperparámetros de la combinación lineal que construye el target.

[datos] Evidencia que motivó la entrada: (1) el mismo juego de voltajes dio
2, 4, 5, 6, 8, 9 y 10 hits en corridas idénticas de SIMION (haz sin semilla
fija) — los hits son ruido de disparo a ~2% de transmisión; (2) en el
barrido de 37 corridas del 2026-07-04, los hits no rankearon nada pero las
features sí mostraron estructura limpia (V6 +15V comprime el grupo a RMS
(2.4, 5.5) mm; la asimetría del par + con +15V mata el haz completo);
(3) descubrimiento clave: SIMION ya registraba las velocidades en el
recording (out.txt) — Vx, Vy, Vz, KE y TOF por ion en cada evento; el
parser viejo (get_positions) las descartaba. beam_characterization.py las
extrae ahora sin cambiar nada de SIMION. Primera medición nueva: el grupo
que llega al detector diverge a 79 mrad en x contra 164 mrad en y — la
óptica de salida abre el haz en y al doble de tasa que en x, señal
invisible para el conteo de hits.

## 2026-07-05 — Reorganización del espacio de trabajo y fachada única

La intención es establecer un espacio limpio de trabajo donde se pueda
desarrollar el gemelo digital sin disruptir el orden esperado de un
software desarrollado. De esta manera se tiene de manera clara qué misión
tiene cada archivo y cómo correr el gemelo digital.

[datos] Antes: 15 módulos Python activos mezclados en la raíz con bases de
datos, documentos y un out.txt de 24 MB; tres problemas de dependencias
medidos (import invertido física→calificador, orchestrator como
dios-módulo del que 4 herramientas importaban, y recarga de ~40 s de
campos base en cada herramienta). Después: raíz con solo los 9 módulos
mínimos + gemelo.py (fachada única con .predecir/.evaluar/.entrenar/
.mejor), 9 herramientas en tools/, estudios en studies/, resultados en
outputs/, zona libre en playpen/. Verificado de punta a punta: la fachada
encontró el mejor global (10 hits), lo predijo en RK4 (51.5 mm) y lo voló
en SIMION real (19.9 mm, 2 hits — ruido de disparo ya documentado). De
paso se detectó una DB vacía de un arranque abortado y que solo 2 de 200
partículas del RK4 terminan sin marca de pared (sobre-marcado que las
features nuevas van a corregir).

## 2026-07-05 — Un solo analizador de beam para SIMION y RK4

Nos dimos cuenta que SIMION y RK4 están midiendo cosas diferentes, y por
eso el resultado entre ambas simulaciones daba discrepancia en la pequeña
escala. Ahora, para resolver eso se diseñó un solo analizador de beam, el
cual es capaz de medir los mismos parámetros que se miden en este momento
en beam_progress_rank, y agrega Twiss, halo, kurtosis y residuo de matriz
de transporte. En este momento entonces los resultados de SIMION y RK4
son calificados con el mismo analizador, lo que produce comparaciones más
sensatas.

[datos] caracterizador.py: ~22 features por corrida, mismas primitivas de
distancia/hits que optimizer.count_hits. Tests sintéticos en playpen:
Twiss recuperado con <1.2% de error; residuo de transporte ~1e-13 para un
mapa lineal exacto y 1.1 mm al inyectar una aberración cuadrática.
Primera medición sobre el vuelo real del mejor config: alfa de Twiss
POSITIVO en ambos planos (el haz llega todavía convergiendo: el foco cae
más allá del detector) y el plano y es el limitante por triple vía —
emitancia 1090 vs 149 mm·mrad (7x), residuo de aberración 8.3 vs 3.2 mm
(2.6x), divergencia 175 vs 74 mrad (2.4x). Advertencia vigente: del lado
RK4 solo 2/200 partículas quedan "limpias" (sobre-marcado de pared), así
que las features RK4 aún no son comparables en la práctica; el dataset de
calibración (Task B) debe alimentarse por ahora del lado SIMION, que el
registro de la fachada guarda automáticamente.
