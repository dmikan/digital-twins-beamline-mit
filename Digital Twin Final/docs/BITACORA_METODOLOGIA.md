# Bitácora de metodología — en palabras de Julián

Registro de decisiones para el informe final (deadline: 10 de julio).
Regla de trabajo: cada vez que hagamos un cambio o tomemos una decisión,
Julián escribe acá **en sus propias palabras y en español** qué estamos
haciendo y por qué. Claude pregunta, Julián redacta, Claude solo pega la
entrada tal cual (sin cambiar el lenguaje) y agrega los datos duros de
soporte debajo, marcados como [datos].

Organización: por **tema** (Física, Arreglo de código, Optimización,
Evaluación), no por fecha — cada entrada mantiene su fecha en el título
para que la línea de tiempo se pueda reconstruir dentro de cada tema.

Formato de cada entrada:

```
### AAAA-MM-DD — <título corto>
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
- [ ] Bug crítico del objetivo J_v2 (rankeaba 0-hits sobre el récord) y su fix.
- [ ] Fix PRUNED vs BAD_SCORE en el loop de Optuna.
- [ ] Limpieza/archivado de studies/ (¿entra o es housekeeping menor a omitir? pendiente de confirmar).
- [ ] Consolidación de código raíz (15 módulos → 5).
- [ ] Evaluación formal: tarea Dirección + tarea Consigna Inversa.
- [ ] Radio de confianza del gemelo (transmisión vs. centroide).
- [ ] Barrido de ángulo — límite físico de transmisión.
- [ ] Descubrimiento de "cuencas" (basins): TPE vs GP convergen a óptimos
      de polaridad opuesta.

FUERA DE SCOPE (decisión 2026-07-08): la cadena de colimación
pre-cuádruple (→ récord 43 hits) y el récord de 61 hits (barrido de
placas V15/V18) NO van en la bitácora — fueron caza manual de voltajes
vía barridos/arbitraje directo en SIMION, no algo que el gemelo/
optimizador haya encontrado por sí solo. La prioridad de la bitácora es
documentar el entrenamiento del gemelo, no atajos manuales que lo
bypasean.

---

<!-- Las entradas nuevas van al final de su sección temática. -->

## Física

### 2026-07-03 — El gemelo empieza a coincidir con SIMION: fix de unidades y primera predicción exitosa

Empezando el viernes, después de obtener un hit el trabajo se enfocó en
desarrollar la implementación actual que prometía pero claramente estaba
incompleta. El diagnóstico en el momento era que el modelo predecía
correctamente muy malos candidatos, pero sus predicciones favorecidas no
correspondían a los resultados en SIMION. Una breve investigación arrojó
el hecho de que la fórmula de aceleración no estaba tomando en cuenta que
las unidades del simulador eran V/mm y mm debido a la escala de SIMION.
Una vez se solucionó esto, los resultados de SIMION y RK4 empezaron a
coincidir.

Una posible solución que se consideró para mejorar el modelo fue
entregarle una mejor semilla inicial para iniciar el study (el cual en
este momento se estaba trabajando con TPE). Como RK4 ya correspondía a
SIMION, se hizo un scan del bender para obtener una respuesta de señal en
el RK4. Efectivamente, con esta nueva semilla nuestro gemelo fue capaz de
obtener 2 hits, siendo esta la primera predicción exitosa del gemelo.

[datos] El bug: `aceleración = q·E/m` es válida solo en SI (E en V/m,
metros); el código tomaba E directo en V/mm con posiciones en mm sin
convertir, subescalando cada fuerza eléctrica por 1,000,000x (V/mm→V/m
es ×1000, m/s²→mm/s² es otro ×1000). Fix: constante
E_VMM_TO_ACCEL_MM=1e6 en RK4_sim_basis_batch.py. Coincidencia verificada
post-fix: con el bender en cero voltios, el RK4 corregido manda el haz
derecho contra la pared en x~0, z~80 — coincide con los splats reales de
SIMION en z=1-174mm (antes del fix, RK4 no reproducía esto en absoluto;
las trayectorias eran idénticas a un decimal sin importar el voltaje).
DT bajado de 5e-8 a 1e-8 (el haz sale a ~470eV reales, no 15eV — el paso
viejo daba ~2.9mm/paso, demasiado grueso). El scan del bender
(bender_field_analysis.py) localizó los electrodos 9-12 en el plano x-z
y derivó el patrón de pares diagonales (9,12)/(10,11); el punto de
partida resultante (V3=-300, V6=-300, V9=-455, V10=+455, V11=+455,
V12=-455, V15=0, V18=0) dio en RK4 un giro real de 90 grados: 50% de
partículas a la caja del detector, error medio 5.6mm (contra 0% /
~325mm de error para todo lo previo al fix). Confirmación en SIMION real:
el trial 42 (perturbación gaussiana del punto derivado) anotó 2 hits
reales — el primer resultado no-cero del proyecto tras 26+ corridas
previas, todas en 0.

### 2026-07-04 — Ampliar la información extraída de cada corrida de SIMION

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

### 2026-07-05 — Un solo analizador de beam para SIMION y RK4

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

### 2026-07-06 — De la geometría por STL al mapa de paredes extraído del PA

Sobre la geometría, luego de aprovechar la geometría inscrita en los
archivos STL, se presentó una primera aproximación de esta dentro del
RK4. Sin embargo, nos dimos cuenta de una discrepancia entre los marcos
de referencia de los archivos STL y las coordenadas de SIMION. Se
planteó una primera modificación manual corrigiendo el desfase con
medidas manuales de los electrodos desde el GUI de SIMION; esto, sin
embargo, no fue una corrección completa, igualmente no teníamos una
referencia clara de los electrodos en las coordenadas necesarias. Con
esto en mente, aprovechamos la información contenida en el archivo .PA0,
el cual conoce si un punto en la grilla cae dentro de un metal. Con esta
información, de manera similar a como se hizo con el potencial, se
construyó el mapa de la geometría directamente desde SIMION, de esta
manera alineando las coordenadas y solucionando el residuo de alineación
encontrado.

[datos] Primera aproximación (previa a esta sesión): electrode_geometry.py
ajustaba una transformación rígida (rotación + traslación) a los STL vía
mínimos cuadrados, usando 6 puntos de referencia leídos a mano del cursor
del GUI de SIMION (posición de fuente, detector, y los 4 centros de
electrodo del cuadrupolo). Residuo medido el 2026-07-06 tras construir la
referencia independiente (máscara de metal del PA): ~25mm de desfase
promedio en los centroides de los electrodos 9-12 del cuadrupolo, y
6.6mm de mediana (16.1mm p90) entre la superficie STL y el metal real.
Ese residuo era la causa directa del punto ciego del RK4 en el
cuadrupolo (predecía ~0% de transmisión para configs que en SIMION
transmitían ~78%). Solución: `pa:point(x,y,z)` de SIMION devuelve a la
vez el potencial y si el punto cae dentro de un electrodo sólido
(es_metal) — se barrió la cámara en grilla regular (2mm global, 1mm en
la zona del cuadrupolo) preguntando es_metal en cada punto
(extract_PA_quad.lua), y luego, usando fast_adjust() para fijar en 1.0
los electrodos a conservar como pared y en 0.0 los excluidos (1=fuente,
2=tubo, 19=detector), se generaron máscaras donde "es pared" =
es_metal AND potencial>0.5 (extract_wall_masks.lua) — la exclusión queda
codificada en el propio PA, sin ninguna dependencia del STL. ParedesPA
(physics.py) carga esos puntos y arma un cKDTree para consulta de
distancia.

Validación y decisión de promoción (playpen/comparar_porteros.py, mismo
dataset y misma corrida para ambas físicas): en la población legada de
configs archivados quedó un EMPATE como portero de ranking (Spearman
+0.294 física nueva vs +0.303 física vieja; 6/14 vs 5/14 hitters en el
top-30), pero en la familia del mejor récord la física nueva DOMINÓ por
completo — predijo 388/395 partículas limpias contra 368/384 medidas en
SIMION real, mientras la física vieja predecía CERO llegada y 85% de
choques contra pared fantasma para esos mismos voltajes. Con ese
resultado (empate donde no importaba, victoria total donde sí) se
decidió promover la física nueva (CampoDual/ParedesPA) a producción.

### 2026-07-06 — Margen de contacto con pared (0.5mm) y decisión de resolución de malla (2.5mm)

Agregar sobre paredes PA: una vez se agregaron las paredes directamente
desde .PA0, se corrigió un margen de 0.5 mm para comprobar si una
partícula chocó con una pared.

Durante el desarrollo del proyecto se pensó en la idea de mejorar la
capacidad del RK4 de identificar el efecto del campo en las trayectorias,
con la intención de que de esta manera RK4 pudiera identificar de manera
positiva un mayor número de trayectorias reales. Esto, sin embargo, no
demostró ninguna mejora en la precisión de los resultados en el bender,
y además incrementó considerablemente el tiempo de cómputo. Por eso se
decidió quedarse con la grilla de 2.5 mm.

[datos] Margen de contacto: se barrieron tres valores (1.5mm, 1.0mm,
0.5mm) sobre los 5 casos canónicos (vivos y muertos conocidos) y el
ranking de los configs archivados. A 0.5mm el ranking mejoró frente a
1.5mm (Spearman +0.214 vs +0.077) y los casos vivos se acercaron mucho
más a SIMION real — el RECORD predijo 265 partículas limpias a 0.5mm
contra solo 20 a 1.5mm (SIMION real: 384). Se adoptó 0.5mm como margen
de producción (MARGEN_FINO en physics.py): con paredes reales del PA ya
no hace falta el colchón de seguridad de 1.5mm que existía para
compensar el desalineamiento del STL (ese margen viejo queda como
fallback, MARGEN_STL, solo si faltan los datos de la malla nueva).

Resolución de malla: se probó llevar las cajas finas de campo/pared de
2.5mm a 1mm (la resolución nativa del PA), esperando que RK4 resolviera
mejor el campo fuerte del cuadrupolo/lentes. Comparación controlada
(mismo dataset, misma corrida, 2026-07-07,
playpen/comparar_resolucion.py) sobre 70 configs archivados: Spearman
+0.168 (1mm) vs +0.160 (2.5mm) — estadísticamente el mismo ranking, sin
mejora real. Costo: la malla de 1mm tiene ~15x más puntos por caja
(ej. la caja del cuadrupolo pasa de ~105mil a ~1.58 millones de puntos
por electrodo), obligando a bajar el tamaño de chunk de 32 a 8 configs
por la memoria del campo (E_batch), con el consiguiente aumento del
tiempo de integración por config (~4.25 s/config a 1mm vs ~2.88 s/config
a 2.5mm, medido en el mismo chunk de screening). Decisión: producción
se queda en cajas finas de 2.5mm (CampoFino, physics.py) — la malla de
1mm quedó respaldada en basis_quad/backup_1mm/ por si hace falta
retomarla.

*(nota técnica, no dictada por Julián — agregada a pedido explícito,
documenta el estado actual del código, no una decisión de esa fecha)*

Por cada partícula, se consultan sus posiciones grabadas durante el
vuelo (con paso `wall_check_stride=3` — cada 3er paso guardado, más
SIEMPRE el primero y el último activo, nunca los puntos medios entre
pasos en el screening: `wall_check_midpoints=False`) contra
`wall_index.distance(punto)` — una consulta a un cKDTree de puntos de
metal REAL extraídos del PA (ParedesPA, ver entrada anterior), no del
STL. Un punto cuenta como "toque de pared" si
`distancia <= wall_hit_margin` (hoy dinámico: 0.5mm con la física nueva,
1.5mm si cae al fallback STL). La consulta solo corre en la ventana
"todavía en vuelo" de cada partícula (antes de resolverse como
llegada/perdida). `hit_wall` por partícula = True si CUALQUIERA de sus
puntos muestreados cae dentro del margen de un punto de metal;
`wall_hit_fraction` (=$f_{\text{pared}}$) es el promedio de `hit_wall`
sobre las N partículas de ese config. Importante: tocar pared NO mata la
trayectoria en producción (`terminate_on_wall_hit=False`) — la partícula
se sigue integrando y contando normalmente para el resto del puntaje;
es solo una bandera para el término de penalización. Terminar al tocar
pared se probó y se descartó (ver entrada de Optimización del 03-07:
perdía contra la penalización blanda, 7-9/61 vs 17/61 hitters en top-30).

---

## Arreglo de código

### 2026-07-05 — Reorganización del espacio de trabajo y fachada única

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

---

## Optimización

### 2026-07-03 — De contar hits a la posición del 10%: construyendo el objetivo denso del RK4

El desarrollo de la función objetivo del RK4 se basó en el avance
progresivo del proyecto. Una vez teníamos funcionando el modelo, nos
dimos cuenta que como no teníamos hits, nuestro modelo nunca iba a
arrojar mejores predicciones (el gemelo está en un espacio plano), lo
que nos llevó a armar el ranking en términos de la posición media del
rayo al detector, que luego se transformó en la posición del 10% pues la
señal de todas las partículas disminuye la diferenciación de buenos runs
y malos runs. Se incluye una recompensa para beams que presentaban una
buena fracción de partículas que efectivamente llegan al sensor.

[datos] El problema de partida (documentado como "el espacio es plano"):
con el conteo de hits puro, 0 hits en TODAS las corridas (RK4 y SIMION
real) hacía indistinguibles voltajes que diferían hasta en 1800V en un
electrodo — cero gradiente para que el sampler aprenda algo. Primer
intento: distancia MEDIA de splat de todas las partículas al detector —
insuficiente, porque el grueso del haz revienta temprano contra el
einzel (~324mm afuera) y dos configs de 0 hits diferían apenas ~3.5mm en
esa media, ruido indistinguible de señal. Solución: usar solo la
distancia media del 10% de partículas MÁS CERCANO (SPLAT_TOP_FRACTION),
"qué tan cerca llega el borde de ataque del haz" — ahí sí discrimina.
Problema nuevo que esto destapó (saturación): un config que ya transmite
tiene sus 50 partículas más cercanas splasheando DENTRO del detector, así
que la distancia del 10% da exactamente 0.0 tanto para 5 hits como para
100 — el término se queda sin gradiente justo cuando la transmisión es
lo que hay que optimizar. Fix: sumar un término de recompensa por
transmisión, HIT_TERM_SCALE_MM=20 * (1 - fracción_que_llega) — configs
lejanos siguen rankeados por distancia (+20mm constante), configs
saturados por cuánto del haz realmente entra (5/500→+19.8, 100/500→+16,
transmisión total→+0). Esta es la forma que sigue vigente hoy en
rk4_score_chunk (optimizer.py), con el término de pared y el cinemático
agregados después (ver entrada de Física del 06-07 y la discusión de
pesos).

### 2026-07-06 — Bug crítico en el objetivo J_v2: rankeaba haces sin hits por encima del récord

En la optimización del gemelo nos dimos cuenta que constantemente, la
función objetivo del gemelo estaba rankeando incorrectamente como altos
beams que no registraban hits. Esto sucedía pues la transmisión la
estábamos midiendo con una ecuación lineal; el problema con esto es que
este término no responde de manera significativa a cambios de
transmisión, ahogando la señal por los otros parámetros. Por eso
cambiamos la medición de transmisión por una función racional,
entregándole más peso a los primeros hits y disminuyendo continuamente
el peso del término por parámetros que determinan la forma del beam.

[datos] El objetivo J_v2 combina transmisión + términos de forma del haz
(offset, halo, kurtosis, colimación, Twiss, cuerpo). El término de
transmisión era `1 - hits/500` (lineal): entre 0 y 47 hits (un récord
real) solo se movía de 1.0 a 0.906 — apenas 9% de su rango — mientras
los términos de forma barrían su rango completo [0,1] con peso
comparable. Medido sobre 47 configs reales del registro:
Spearman(hits, J) = **+0.020** (sin correlación, y de signo equivocado
ya que J se minimiza) — en la práctica, `tw.mejor()` devolvía un config
de 0 hits en vez del récord conocido de 43. Fix: transmisión pasó a
`1 / (1 + hits/H0)` (racional, empinada cerca de 0 — los primeros hits
valen mucho más), con H0_HITS recalibrado de 3 a **15** (H0=3 saturaba
demasiado rápido y perdía capacidad de distinguir entre configs de
30-60 hits, el rango donde ya estaba trabajando el proyecto) y el peso
del término subido de 0.40 a 0.50. Verificado: Spearman(hits, J) pasó de
+0.020 a **-0.87**. Se agregó además `mejor(por="hits")` en gemelo.py
como método separado de `mejor()` (por J) — incluso corregido, J puede
seguir prefiriendo un haz más apretado con algunos hits menos sobre uno
más ancho con más hits, un trade-off de calidad-vs-cantidad deliberado,
no un bug.

### 2026-07-07 — TPE como motor principal, y la siembra de la mejor semilla del GP dentro del muestreo del RK4

En optimización es importante mencionar que la mayoría del código se
estaba trabajando con TPE, el cual no explota la función del GP. Por la
mayoría del desarrollo del proyecto se trabajó con este modelo debido a
la robustez de la implementación junto con el RK4; esta combinación nos
permitió encontrar nuestras primeras soluciones. Para aprovechar el
máximo de ambas funciones, se integró dentro de los candidatos sugeridos
del RK4 siempre la mejor semilla del GP, de tal manera exploramos el
espacio mientras que explotamos mejores soluciones.

[datos] TPE (Tree-structured Parzen Estimator, el sampler por defecto de
Optuna) fue el motor de la mayoría de las corridas del proyecto,
combinado con el filtrado barato de RK4 — esta combinación produjo las
primeras soluciones reales del gemelo (los primeros hits, ver entradas
de Física del 03-07). El mecanismo de integración del GP: en vez de
correr el GP como sampler independiente (donde su `sample_independent()`
para el muestreo barato ignora por completo el modelo GP real), se pagan
sus consultas costosas (`ask()`/`sample_relative()`, ~100-300ms y
creciendo con el historial) solo `n_gp_seeds` veces por iteración
(default 5), y esas semillas informadas se mezclan DENTRO del mismo lote
de candidatos que el RK4 screening evalúa junto con perturbaciones
baratas alrededor de ellas — así se explora el espacio amplio (TPE/RK4)
mientras se explota el mejor punto que el GP cree encontrar.

Evidencia en fig_convergencia_gemelo.png (outputs/report_figures/):
durante las primeras ~70-80 evaluaciones ambos motores corren muy
parejos, entrelazándose (TPE incluso adelante buena parte del tramo);
recién cerca de la evaluación #80 el estudio GP-seeded encuentra un
mínimo bastante mejor (J≈0.21) y lo sostiene, mientras TPE se estanca en
J≈0.23. En términos de hits reales esto corresponde a un salto de
récord bien superior a lo que TPE solo había encontrado en ese momento.

Matiz honesto (ya medido en outputs/comparacion_TPE_GP/comparacion_TPE_vs_GP.md,
que no debe perderse): a IGUAL número de evaluaciones (60 cada uno) el
panorama se empareja — el pico de TPE (49 hits) fue incluso mayor que
el de GP en ese mismo tramo (42 hits), y el mejor J casi empatado
(0.283 vs 0.281). La ventaja grande del GP a presupuesto completo (133
vs 49 hits) viene en buena parte de que corrió más del doble de
evaluaciones (150 vs 60), no de una superioridad garantizada del sampler
por evaluación — el GP paga su exploración tarde y recién despega con
presupuesto extra.

Quiero que quede claro que en este momento, aunque ambos modelos
muestran comportamiento similar, la ventaja del GP brilla cuando le
entreguemos un poco más de información sobre la estructura de la
solución, al incluir en su construcción la correlación entre los
electrodos del bender.

[datos] Nota: esto es una dirección de diseño propuesta, todavía NO
implementada ni medida — no confundir con los resultados anteriores.
La ventaja teórica del GP frente a TPE es precisamente modelar
CORRELACIONES entre variables (su kernel de covarianza), algo que TPE no
hace (sus KDEs son por parámetro, independientes entre sí). Hoy el GP
trata los 8 electrodos optimizables como independientes, exactamente
igual que TPE — no se le está dando la información que sabemos que
existe: desde la entrada de Física del 03-07, el bender (electrodos
9-12) responde a PARES DIAGONALES correlacionados (9,12) y (10,11), no a
4 voltajes independientes. Si esa correlación se codifica explícitamente
en la construcción del GP (p.ej. un kernel que la refleje, o
reparametrizar a ~2 parámetros de "fuerza de deflexión" en vez de 4
voltajes crudos — la opción #2 que quedó anotada y sin implementar desde
el 03-07), el GP tendría la estructura que necesita para superar a TPE
de forma consistente, no solo cuando tiene el doble de presupuesto.
Queda como próximo paso a probar, no como resultado ya verificado.

### 2026-07-08 — Cuencas (basins): el espacio de soluciones es fino y hay al menos dos óptimos de polaridad opuesta

Para terminar, debido a lo fino que es el espacio de soluciones que
logran causar la deflexión del rayo, son muy sensibles y difíciles de
conseguir aleatoriamente. De tal manera, nos dimos cuenta de la
existencia de por lo menos dos cuencas donde existen soluciones con
transmisión pero con la dirección de los voltajes opuestas. Se cree que
con la implementación de la física dentro del modelo GP este pueda
discernir entre cuál de estas cuencas es la correcta.

[datos] Comparando los mejores configs de dos corridas GP-seeded
distintas (mismo mecanismo de descubrimiento: ambas encontraron su mejor
trial en la iteración 3, evaluaciones #78 y #80, vía rk4_rank 8 y 10 —
no fue casualidad de una corrida particular):

| | corrida A | corrida B |
|---|---|---|
| mejor resultado | 93 hits | **133 hits** |
| V3 | -504 | **+296** |
| V6 | -126 | **+359** |
| V9 | -415 | **+117** |
| V12 | -220 | -606 |
| V18 | -392 | **-5** |

Distancia entre ambos mejores configs: **|ΔV| = 1241 V** (norma L2) — no
es la misma solución refinada dos veces, son regiones distintas del
espacio de 8 dimensiones, con **signo invertido** en varios electrodos
clave (V3, V6, V9). La corrida A quedó atrapada en su cuenca durante las
100 evaluaciones completas sin escapar nunca hacia la cuenca mejor,
aunque el mecanismo de búsqueda (GP-seeded + RK4 + Optuna) era
idéntico — la diferencia fue el punto de arranque, no el algoritmo.
Esto es consistente con la entrada anterior: ni TPE ni GP (tal como
están implementados hoy, sin conocer la estructura física del problema)
tienen manera de saber que existe una cuenca mejor del otro lado del
espacio de voltajes; solo exploran localmente alrededor de donde ya
empezaron. La hipótesis de que la física (la correlación de pares
diagonales del bender) podría ayudar al GP a discernir entre cuencas
queda como trabajo futuro, no verificada todavía.

*(pendiente de redactar — ver lista de backfill arriba: tarea Dirección,
tarea Consigna Inversa, radio de confianza del gemelo, barrido de ángulo
para el límite físico de transmisión.)*
