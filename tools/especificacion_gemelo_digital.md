# Especificación Técnica del Gemelo Digital y Recolección de Datos

## 1. Cantidad Elegida para Modelar y Justificación

El objetivo del gemelo digital no consiste en modelar directamente el campo eléctrico del sistema ni las trayectorias individuales de las partículas, sino en aproximar una función objetivo escalar que representa la calidad global del haz obtenido para una configuración determinada de voltajes. Esta cantidad corresponde a la función $J_{v2.4}$, la cual integra en un único valor las características más relevantes del haz y permite comparar objetivamente distintas configuraciones de operación.

La elección de una única función objetivo presenta varias ventajas:
* **Reducción de la dimensionalidad:** Reduce el problema de optimización a minimizar un escalar en lugar de optimizar simultáneamente múltiples variables físicas.
* **Eficiencia computacional:** La evaluación de esta función puede realizarse mediante el modelo RK4 del gemelo digital, evitando ejecutar una simulación completa en SIMION para cada candidato y reduciendo considerablemente el costo computacional.

La función objetivo se calcula a partir de las características extraídas del haz mediante el módulo `caracterizador.py`, el cual analiza las trayectorias finales de todas las partículas y calcula métricas como:
* Número de partículas que alcanzan el detector.
* Distancia media de las partículas al detector.
* *Offset* del centro del haz.
* Tamaño del haz ($\sigma_x$ y $\sigma_y$).
* Divergencia angular.
* Parámetros de Twiss.
* Emitancias.
* Fracción de halo.
* Kurtosis.
* Residuos del transporte.

Cada una de estas cantidades representa un aspecto diferente de la calidad del haz. Posteriormente se combinan mediante una suma ponderada que produce el valor final de $J_{v2.4}$.

La versión final del trabajo utiliza la función $J_{v2.4}$, que incorpora una modificación importante respecto a versiones anteriores. Inicialmente la transmisión del haz se evaluaba mediante un término saturante, el cual dejaba de distinguir adecuadamente configuraciones con un número elevado de impactos sobre el detector. Para solucionar este inconveniente se añadió un término lineal de transmisión:

$$\lambda \left(1 - rac{	ext{hits}}{N_{	ext{ref}}}
ight)$$

utilizando $\lambda = 2$. Este término aumenta la importancia del número de partículas detectadas durante la optimización y permite diferenciar correctamente soluciones con diferencias relativamente pequeñas en la transmisión, sin alterar el refinamiento local proporcionado por el resto de métricas.

Finalmente, la función objetivo se normaliza para mantener aproximadamente el rango $[0,1]$, conservando la interpretación de que valores menores corresponden a mejores configuraciones del sistema.

---

## 2. Recolección de Datos

Los datos utilizados para entrenar el modelo fueron obtenidos mediante un esquema híbrido que combina simulaciones físicas rápidas con simulaciones de alta fidelidad.

Cada candidato corresponde a una configuración de voltajes aplicada sobre los electrodos libres del sistema. Inicialmente dicha configuración es evaluada mediante el modelo RK4 implementado dentro del gemelo digital, el cual integra las ecuaciones de movimiento utilizando los campos eléctricos previamente calculados mediante superposición de funciones base. Esta evaluación proporciona una estimación rápida de la calidad del haz y permite descartar configuraciones claramente desfavorables.

Las configuraciones más prometedoras son posteriormente evaluadas mediante SIMION, que constituye el modelo físico de referencia. Los resultados obtenidos se utilizan tanto para validar el gemelo digital como para continuar alimentando el proceso de optimización.

Todas las evaluaciones quedan almacenadas automáticamente mediante **Optuna**, registrando para cada ensayo:
* Voltajes aplicados.
* Valor de la función objetivo.
* Número de impactos sobre el detector.
* Características físicas del haz.
* Información auxiliar utilizada posteriormente por el Proceso Gaussiano.

En la versión final del trabajo se empleó una reparametrización del cuadrupolo, reduciendo el número de variables independientes del problema. En lugar de optimizar directamente los cuatro voltajes de los electrodos del cuadrupolo ($V_9$ a $V_{12}$), éstos se describen mediante tres parámetros físicos:
* **A:** desplazamiento común (*offset*) del cuadrupolo.
* **B:** intensidad principal de flexión.
* **C:** término de asimetría horizontal.

Los voltajes físicos se obtienen mediante las relaciones:

$$V_9 = A + B + C$$
$$V_{10} = A - B$$
$$V_{11} = A - B$$
$$V_{12} = A + B - C$$

Esta parametrización reduce la dimensionalidad efectiva del problema sin perder la estructura física del sistema, facilitando el aprendizaje del modelo probabilístico.


---

## 3. El Modelo: Arquitectura, Hiperparámetros y Validación

### 3.1 Arquitectura General
El diseño del gemelo digital se fundamenta en estructurar un modelo de simulación alternativo que actúa como filtro de bajo costo computacional frente a los análisis detallados en el entorno de SIMION. Para simplificar el control de este flujo, el sistema consolida todas sus operaciones en la clase fachada `GemeloDigital` en `gemelo.py`, la cual actúa como punto de acceso central para coordinar el motor de física, el caracterizador del haz de iones y el lazo de optimización.

La estrategia de búsqueda opera mediante un lazo de *screening* en dos etapas secuenciales:
1. **Primera Etapa (Económica):** El módulo de optimización en `optimizer.py` genera un lote amplio de configuraciones candidatas (habitualmente 239 conjuntos de voltajes) que se envían al motor de física de `physics.py`. Este motor propaga las partículas en paralelo y estima la viabilidad geométrica del haz de cada candidato.
2. **Segunda Etapa (De Alta Fidelidad):** El sistema selecciona las configuraciones más prometedoras (el top 10 de la evaluación anterior) y las ejecuta en el entorno real de SIMION. Los datos de impactos (*hits*) y perfiles obtenidos en SIMION se devuelven al optimizador para actualizar el Proceso Gaussiano (GP) que orienta las sugerencias de la siguiente iteración del lazo.

**Justificación del costo de ejecución:** Cada corrida en SIMION consume cerca de $5.77 	ext{ s}$ de tiempo de pared, retraso provocado principalmente por el arranque de `fastadj` y la lectura en disco del archivo de potenciales de $290 	ext{ MB}$ [C1]. Dado que la zona útil de voltajes que transmite partículas representa una fracción extremadamente delgada del espacio de búsqueda de ocho dimensiones —una parte en mil millones ($1	imes 10^{-9}$) [C2]—, una búsqueda directa sobre SIMION consumiría semanas de cálculo. El motor de física intermedio resuelve esta limitación al evaluar cada candidato en apenas $0.25 	ext{ s}$, una relación de velocidad de $\sim 23	imes$ que permite filtrar lotes masivos de 239 candidatos en el mismo tiempo de pared en que SIMION resolvería solo 10 [C1].

### 3.2 El Motor de Física
El motor de física modela la trayectoria del haz de iones resolviendo numéricamente las fuerzas electrostáticas locales en cada paso temporal. La simulación se organiza en cuatro capas de implementación:

* **Superposición de potenciales:** En lugar de calcular la física del campo para cada candidato desde cero, se explota el carácter lineal de la ecuación de Laplace: el campo eléctrico resultante de aplicar cualquier combinación de voltajes se obtiene mediante la combinación lineal de 19 campos base de los electrodos, extraídos de SIMION una sola vez. En el módulo `physics.py`, estos campos se combinan mediante un producto tensorial matricial directo provisto por `numpy`. El campo de fuerza local se obtiene resolviendo los gradientes tridimensionales negativos sobre la grilla de potenciales combinada.
* **Enrutamiento de grillas compuestas:** Para no saturar la memoria con un mapa uniforme de alta resolución, se estructura una grilla compuesta en la clase `CampoFino` de `physics.py`. Esta grilla superpone cajas locales de alta precisión (el cuadrupolo a $1.0 	ext{ mm}$ de espaciado, y los canales de colimación c1 y c2 a $2.5 	ext{ mm}$ de paso) sobre una grilla global más gruesa de $2.0 	ext{ mm}$. La consulta de posiciones utiliza un esquema de prioridad en el que cada coordenada es resuelta por la caja fina más específica que la contiene, y solo los puntos residuales caen al mapa global. Esta técnica reduce la huella de memoria en unas $15	imes$ en comparación con un mapa homogéneo de $1.0 	ext{ mm}$, manteniendo la misma precisión y correlación de ranking de salida. La interpolación espacial continua se vectoriza mediante `scipy.ndimage`, evitando bucles en Python.
* **Integración de trayectorias:** La física de la trayectoria se propaga en un solo paso matricial para todo el lote de partículas y candidatos, mediante el método numérico de Runge-Kutta de cuarto orden (RK4) implementado en la clase `BatchRK4Integrator` de `physics.py`. Las ecuaciones de actualización integran, en cada paso temporal, la posición y la velocidad de los iones a partir de las fuerzas electrostáticas locales. La aceleración de los iones se obtiene a partir de su carga, su masa y el campo eléctrico interpolado, multiplicada por un factor de escala constante para convertir las unidades de campo ($	ext{V/mm}$) y masa a unidades físicamente coherentes de $	ext{mm/s}^2$. Con un paso temporal de $1	imes 10^{-8} 	ext{ s}$, el avance espacial de las partículas es de aproximadamente $0.6 	ext{ mm}$, lo que permite resolver la trayectoria en *gaps* estrechos de forma sub-milimétrica.
* **Geometría de colisiones:** Para determinar qué partículas chocan contra las paredes sólidas, se extrae una máscara metálica directamente de la grilla de potenciales del archivo `.PA0` de SIMION (donde los puntos metálicos activos se definen por potencial y tipo de electrodo), lo que elimina cualquier desfase geométrico. En el código de `ParedesPA` de `physics.py`, estos vóxeles metálicos se estructuran en un árbol espacial para búsquedas rápidas de vecinos más cercanos. Durante la integración, el módulo consulta la distancia de cada ion a la pared más cercana. Para minimizar el costo de cómputo, el chequeo de colisiones se ejecuta cada 3 pasos de integración, logrando una reducción de costo de $\sim 4.2	imes$ con la misma precisión en la clasificación de supervivencia. Si la distancia cae por debajo de un margen límite de contacto de $0.5 	ext{ mm}$, la partícula se clasifica como colisionada.
* **Caracterización del haz y función objetivo:** El caracterizador en `caracterizador.py` mide las propiedades geométricas del haz (distribución espacial, divergencia, emitancia Twiss) en el plano del detector ($z = 390.0 	ext{ mm}$). El éxito de una configuración de voltajes se calcula con la función objetivo ponderada normalizada $J_{v2}$ en el rango $[0, 1]$ más penalizaciones. Esta función combina la transmisión (con una escala de *hits* parametrizada para evitar zonas ciegas), el acercamiento del haz al detector (distancia media del 10% de partículas más cercanas), la fracción de partículas que cruzan el plano, el *offset* radial del haz respecto al centro, el halo del haz, la kurtosis de la distribución, la divergencia angular y la emitancia. Durante el *screening* por RK4 se añade además una penalización lineal proporcional a la fracción de partículas colisionadas contra la pared.

> **Figura 1.** Trayectorias RK4 del gemelo central (*bender7d*, mejor configuración del estudio Search v2.1) en el plano de flexión XZ, con contornos de potencial en $Y = 75 	ext{ mm}$ y la geometría del colimador extraída del archivo PA.

### 3.3 Hiperparámetros
A continuación se detallan los valores seleccionados para los hiperparámetros principales, junto con su justificación:
* **Paso de integración ($dt$):** $1	imes 10^{-8} 	ext{ s}$. Produce un avance de $\sim 0.6 	ext{ mm}$ por paso, lo que permite resolver la trayectoria en *gaps* estrechos de forma sub-milimétrica.
* **Fidelidad de *screening* en dos etapas:** Etapa A, económica (50 partículas, 1500 pasos, semilla fija), para rankear todo el lote; etapa B, más fina (200 partículas, 3000 pasos, semilla distinta), aplicada solo al *top* de la etapa A. Este esquema evita sobreajustar a una única muestra del haz y concentra el costo fino en los candidatos que la etapa A no descarta.
* **Margen de contacto con pared:** $0.5 	ext{ mm}$ sobre la geometría extraída del archivo PA (el margen físicamente correcto para paredes reales).
* **Peso de transmisión en el objetivo ($H_0$):** $15$. Evita saturar antes de los 30–60 *hits* y mantiene la capacidad de distinguir entre esos candidatos. Este término saturante se complementa con un término lineal de transmisión ($\lambda = 2$, re-normalizado) [C13].
* **$n_{	ext{gp\_seeds}} = 5$:** Número de veces por iteración en que se evalúa el modelo real del GP para sugerir candidatos, antes de generar el resto del lote mediante perturbaciones económicas alrededor de ellos, optimizando el tiempo de cómputo.

### 3.4 Validación del Modelo
El criterio adoptado para aceptar cualquier cambio al modelo constó de dos condiciones: primero, que el cambio pudiera reproducir los resultados de la versión anterior (que no afectara lo que ya funcionaba); segundo, que representara una mejora medible frente al original, sin que el costo de implementación o ejecución superara el presupuesto disponible dentro de la hackathon.

Este criterio se aplicó de forma sistemática mediante `validate_rk4_filter.py`, un arnés que reevalúa cualquier versión del filtro RK4 contra el historial completo de resultados reales de SIMION (288 configuraciones con verdad de terreno conocida, sin ejecutar ninguna corrida nueva) y reporta la correlación obtenida y cuántos *hitters* reales caen en su *top-30*. Con este método se determinó, por ejemplo, que terminar la trayectoria de una partícula al tocar la pared —un cambio que parecía más realista por replicar el comportamiento de SIMION— en realidad empeoraba el ranking (7–9 de 61 *hitters* reales en su *top-30*, frente a 16–17 sin esa terminación) [C3], por lo que se descartó pese a la intuición inicial. Con el mismo criterio se aceptó el cambio de interpolador de campo (resultados idénticos, $1.6	ext{–}2.6	imes$ más rápido) y la reducción del chequeo de colisiones a cada 3 pasos en lugar de cada paso (mismo puntaje, $\sim 4.2	imes$ menos costo); revisar cada 5 pasos, en cambio, perdía señal y fue rechazado.

La misma disciplina se aplicó a los cambios de física: al reemplazar la geometría STL por la extraída del archivo PA, la aceptación no se basó en que la nueva geometría “pareciera más correcta”, sino en su desempeño frente a 5 casos canónicos con resultado de SIMION conocido (dos vivos, dos muertos, uno limítrofe) y frente al mismo conjunto de 288 configuraciones usado por `validate_rk4_filter.py`, exigiendo que igualara o superara a la versión anterior antes de promoverla a producción [C10]. De igual forma se evaluó el margen de contacto ($1.5$, $1.0$, $0.5 	ext{ mm}$) y la resolución de la malla de campo ($1 	ext{ mm}$ nativa del PA frente a $2.5 	ext{ mm}$): en el caso de la resolución, la malla más fina no mejoró el ranking pero sí multiplicó el costo de cómputo, por lo que se conservó la opción más económica [C10].

Fuera del filtro, el modelo completo se verificó contra SIMION en regiones fuera del experimento real: para voltajes aleatorios lejos de la cuenca de solución, el RK4 predice que el haz muere lejos del detector ($\sim 325 	ext{ mm}$ en promedio) y SIMION da un resultado equivalente ($\sim 324	ext{–}338 	ext{ mm}$) [C5]. Esta es una verificación económica pero informativa: si el gemelo y la realidad no coinciden en el caso trivial, no resulta razonable confiar en el caso difícil.

### 3.5 Búsqueda Dinámica: Exploración Radial desde Cero
Aun con el filtro RK4 y la reparametrización del *bender*, la evidencia de cuencas dejaba abierto un problema: el punto de arranque determinaba el resultado de la corrida. La cuenca asimétrica de alto rendimiento es un canal estrecho y empinado en el plano de voltajes del *bender*; un optimizador con radio de búsqueda constante y grande salta por encima de ese canal sin registrar *hits*, y queda atrapado en la cuenca simétrica, más ancha pero menos eficiente ($\sim 30 	ext{ hits}$).

El diseño de la solución partió de una revisión crítica del historial: el análisis cruzado de las bases de datos previas reveló que los primeros resultados altos (95 y 84 *hits* en la campaña TES-GP-GP6D) se habían obtenido con los límites de búsqueda restringidos manualmente a la zona óptima —el electrodo V12 estaba acotado a $[-1000, -604] 	ext{ V}$, bloqueando el resto del espacio—, es decir, no correspondían a una búsqueda real desde cero. Al eliminar ese sesgo y abrir las cotas al rango completo de *hardware*, el modelo 8D colapsó a 38 *hits* (Search v1): el volumen de $2000^8$ combinaciones lo dominó. Ese resultado es el punto de partida legítimo frente al cual debe medirse todo lo que sigue.

La solución implementada es un plan de búsqueda dinámica en 6 rondas que automatiza la transición de exploración global a explotación local, con cuatro componentes:
1. **Inicialización insesgada:** Todos los voltajes optimizables arrancan en exactamente $0.0 	ext{ V}$, forzando una exploración radial desde el centro de la línea de haz, sin heredar sesgo de corridas anteriores.
2. **Templado (*annealing*) de perturbaciones:** La desviación estándar de las perturbaciones alrededor de la mejor configuración histórica se reduce geométricamente por ronda, de $500 	ext{ V}$ hasta un piso de refinamiento de $15 	ext{ V}$.
3. **Encogimiento dinámico de límites:** Las cotas locales del *bender* se estrechan alrededor del mejor punto histórico, del 100% del ancho en la ronda 1 al 40% en la ronda 6.
4. **Desacople del espacio del GP:** Para evitar que Optuna degradara a muestreo aleatorio por el cambio de límites en `suggest_float`, el *sampler* consulta siempre límites estáticos en la base de datos, y el encogimiento se aplica exclusivamente en la generación *offline* de perturbaciones que alimenta el filtro RK4.

El costo del *screening* también se administra por etapas mediante el parámetro de penalización cinemática $\kappa$: en las rondas 1 y 2 el cálculo cinemático (aberración, emitancia, Twiss) se desactiva por completo ($\kappa = 0$) y el puntaje se concentra en si los iones cruzan o no el plano, lo que redujo el tiempo de *screening* de $\sim 120 	ext{ s}$ a menos de $10 	ext{ s}$ por ronda; en la ronda 3 se activa con peso moderado ($\kappa = 10$) para orientar la búsqueda hacia el enfoque; y en las rondas 4 a 6 se endurece ($\kappa = 30$) para pulir el *spot*.

El resultado validó el diseño: arrancando en $0.0 	ext{ V}$ y explorando el rango completo de *hardware*, la versión 7D localizó y refinó de forma orgánica el canal de transmisión asimétrico —63 *hits* (trial #53) en 61 trials—, con voltajes seguros para el *hardware* (deflectores prácticamente apagados: $V_{15} = -48 	ext{ V}$, $V_{18} = 0.2 	ext{ V}$). El contraste con la solución 6D de fuerza bruta (113 *hits*, pero con tres electrodos saturados en sus límites de $\pm 1000 	ext{ V}$, con riesgo de arqueo de vacío) es parte de la razón por la que la selección del gemelo central no se decidió únicamente por el pico de *hits*.

> **Figura 2.** Esquema de búsqueda dinámica en 6 rondas: templado geométrico de las perturbaciones (500 a $15 	ext{ V}$), encogimiento de límites del *bender* (100% a 40%) y activación por etapas de la penalización cinemática $\kappa$. Elaboración propia a partir de la implementación en `optimizer.py`.

---

## 6. Metodología

### 6.1 Diagnóstico Inicial y Primera Reformulación del Flujo
La implementación inicial del gemelo digital siguió el flujo básico descrito en el documento de la hackathon, combinando Optuna y SIMION. Esta implementación no presentó dificultades técnicas; el obstáculo principal fue otro: la búsqueda de una solución en un espacio de parámetros prácticamente inexplorado.

Una limitación identificada en el flujo original es que este invoca a SIMION una única vez por candidato, sobre el punto que optimiza la transmisión. Sin embargo, el Proceso Gaussiano (GP) subyacente está diseñado para mejorar el conocimiento de la distribución de la transmisión en función de los candidatos, no para optimizar la forma del haz —que este se doble 90 grados y llegue relativamente colimado—. La magnitud de esta limitación quedó cuantificada al final del proyecto: la banda de voltajes que produce *hits* ocupa apenas $\sim 1	imes 10^{-9}$ del volumen del espacio de búsqueda [C2], y las primeras 26 o más evaluaciones de SIMION con voltajes aleatorios no produjeron ningún *hit* [C8].

A esto se suma que SIMION no está optimizado para ejecutar múltiples simulaciones de voltajes: cada evaluación cuesta $\sim 5.77 	ext{ s}$, dominados no por la física sino por el arranque del proceso y la carga del arreglo de potenciales de $290 	ext{ MB}$ [C1]. Esto implica que una búsqueda exhaustiva sobre SIMION resulta prohibitivamente costosa.

La primera modificación al flujo buscó explotar la física del experimento para explorar el espacio de soluciones de forma más eficiente, mediante un esquema en el que Optuna propone candidatos que son evaluados por un analizador de trayectorias basado en RK4, encargado de rankear la viabilidad del haz antes de invocar a SIMION. En su versión inicial, este analizador solo evaluaba si el punto final de la trayectoria del ion caía dentro del detector. El desarrollo posterior mostró que el analizador debía funcionar como una simulación de trayectoria equivalente a SIMION, no para explorar el espacio real, sino para estimar el potencial de cada candidato de producir un haz físicamente realista.

El simulador RK4 se desarrolló de forma incremental. Las primeras etapas consistieron en construir las clases necesarias para almacenar y procesar los cálculos del RK4, seguidas de la vectorización completa de las funciones mediante `numpy` para permitir su ejecución en paralelo. La función de calificación del haz pasó por varias iteraciones: un puntaje inicial unidimensional a lo largo del eje de viaje (sin distinguir la desviación del haz en los otros dos ejes); luego un puntaje contra el volumen real del detector en 3D, junto con detección de colisiones contra la geometría de los electrodos (basada en un índice de distancias construido desde archivos STL) y penalización de partículas perdidas; posteriormente, la corrección de la escala de la recompensa (de 30 a $150 	ext{ mm}$), tras medir que a las distancias reales de los candidatos el término de recompensa aportaba señal prácticamente nula; y finalmente, la alineación del puntaje del RK4 con el mismo objetivo medido en SIMION —la distancia al detector del 10% más cercano del haz, más un término de transmisión y una penalización por proximidad a paredes—.

### 6.2 Selección del Modelo de Simulación
El analizador tiene como función explorar el espacio de búsqueda con mayor profundidad, con el fin de identificar tanto candidatos claramente inviables como candidatos que producen un haz con las características deseadas; esta función es distinta de la del GP, que optimiza la transmisión y no la forma del haz.

Esto plantea la pregunta de qué tipo de simulador permite investigar el espacio de forma útil sin incurrir en un costo superior al de SIMION. Se evaluaron dos alternativas: modelos estocásticos de aprendizaje (redes neuronales o modelos tipo LLM) y aproximaciones numéricas como RK4.

Se optó por un simulador RK4 por dos razones principales:
1. **Eficiencia en Paralelo:** El método numérico puede optimizarse mediante aritmética matricial vectorizada con `numpy`, permitiendo calcular múltiples trayectorias en paralelo sin recurrir a bucles costosos en Python: en su versión final, el simulador evalúa $\sim 239$ candidatos en el mismo tiempo de pared en que SIMION evalúa $\sim 10$ ($\sim 0.25 	ext{ s}$ por candidato frente a $\sim 5.77 	ext{ s}$, una relación de $\sim 23	imes$) [C1]. Además, el método numérico aprovecha una propiedad física conocida: como la ecuación del potencial es lineal, el campo de cualquier combinación de voltajes es la combinación lineal de 19 campos base extraídos de SIMION una sola vez, por lo que el campo del gemelo es esencialmente exacto por construcción.
2. **Costo de Datos Inviable para Modelos Estadísticos:** Un modelo estocástico habría requerido datos suficientes para construir conjuntos de entrenamiento y prueba. Se estimó ese costo [C2]: dado que un regresor necesita ejemplos positivos (candidatos con transmisión) para aprender algo más que “todo da cero”, y considerando la banda de voltajes donde aparecen los *hits*, un muestreo aleatorio produciría en promedio un positivo cada $\sim 8.6	imes 10^8$ evaluaciones de SIMION. Incluso bajo el supuesto más favorable sobre el tamaño de la cuenca de solución, construir un conjunto de entrenamiento con apenas $\sim 10$ positivos habría costado $\sim 41,000$ evaluaciones —unas 85 veces el presupuesto total de SIMION consumido por el proyecto completo (484 evaluaciones, $\sim 46 	ext{ minutos}$ [C7])—. El modelo estocástico resultaba, en la práctica, inalcanzable dentro del plazo de la hackathon.

### 6.4 Incorporación de Conocimiento Físico del Sistema
Aplicando la misma lógica de explotar el conocimiento físico del experimento, se refinó la comprensión del cuadrupolo como un sistema correlacionado, en lugar de cuatro electrodos independientes. Esto condujo al análisis de campo del *bender* (*bender field analysis*) para obtener una mejor propuesta inicial: se ubicaron los cuatro electrodos del doblador a partir de sus campos base, se midió la deflexión que aporta cada uno por voltio a lo largo de la trayectoria del haz, y se derivó de ahí el patrón correlacionado (pares diagonales de polaridad opuesta) y su intensidad.

El resultado respalda esta decisión: todos los candidatos con *hits* reales encontrados en el proyecto descienden de perturbaciones de ese punto inicial físico, y cada estudio iniciado desde cero encontró su primer *hit* dentro de las primeras 3 a 7 evaluaciones de SIMION [C8], en un espacio donde la búsqueda aleatoria había dado sistemáticamente cero *hits*. Este enfoque fue posteriormente reemplazado en el desarrollo, pero constituyó un paso importante para conocer el espacio de soluciones y comenzar a entrenar gemelos con *hits* en lugar de buscar a ciegas.

> **Figura 3.** Barrido 1D sobre la dirección del patrón físico del cuadrupolo ($V_9..V_{12} = s$ por patrón, RK4 por lotes): la distancia media al detector colapsa de $\sim 325 	ext{ mm}$ a su mínimo cerca de $s = 450 	ext{ V}$. Este punto físico es el origen común de todos los candidatos con *hits* reales del proyecto [C8][C12].

### 6.5 Criterio de Aceptación de Cambios Metodológicos
Estas modificaciones se realizaron aplicando un método sistemático de verificación entre versiones del modelo. Ante cada cambio propuesto se realizaban dos comprobaciones: primero, que la modificación pudiera reproducir los resultados de versiones anteriores ya validadas; segundo, que representara una mejora medible frente al original, sin que el costo o el tiempo de implementación comprometiera la viabilidad del proyecto dentro del plazo de la hackathon.

Ejemplos concretos de este proceso, reproducibles con el código entregado:
* **(i) Cambio de interpolador de campo:** Se aceptó porque produjo resultados idénticos $1.6	ext{–}2.6$ veces más rápido.
* **(ii) Aceleración del chequeo de colisiones:** Revisar cada 3 pasos en lugar de cada paso se aceptó porque reprodujo exactamente los puntajes de la versión exhaustiva con $\sim 4.2	imes$ menos costo, mientras que revisar cada 5 pasos ya perdía señal y fue rechazado.
* **(iii) Terminación de partículas al tocar pared:** Un cambio que parecía correcto por replicar el comportamiento de SIMION se implementó y se descartó, ya que el arnés de validación la evaluó contra 288 configuraciones con resultado de SIMION conocido y mostró que empeoraba el ranking (capturaba 7–9 de 61 *hitters* reales en su *top-30*, frente a 16–17 sin terminación); en su lugar se mantuvo una penalización calibrada, que dejó el filtro en su mejor versión medida (Spearman $+0.458$, 2.7 veces más *hitters* en su *top-30* que el azar [C3]). `validate_rk4_filter.py` reevalúa cualquier versión del filtro contra todo el historial de resultados reales sin gastar una sola corrida de SIMION, y `report_figures.py` regenera las figuras del informe a partir de las bases de datos persistidas.

### 6.6 De la Geometría Aproximada por STL a la Geometría Extraída del Archivo PA
La geometría de colisión usada para detectar choques contra electrodos partió inicialmente de archivos STL, ajustados al marco de referencia de SIMION mediante una transformación rígida (rotación y traslación) calculada por mínimos cuadrados sobre 6 puntos de referencia leídos manualmente del cursor de la interfaz de SIMION. Esta aproximación dejó un residuo medido de $\sim 25 	ext{ mm}$ en los centroides de los 4 electrodos del cuadrupolo [C10], suficiente para invertir el juicio del filtro RK4 en esa zona: predecía transmisión $\sim 0\%$ para candidatos que en SIMION real transmitían $\sim 78\%$ del haz.

La corrección aprovechó que el propio archivo de arreglo de potenciales (`.PA0`) ya identifica, para cualquier punto de la grilla, si este cae dentro de un electrodo sólido —la misma fuente de la que se extrae el campo eléctrico—. Se construyó una máscara de metal directamente desde SIMION (fijando en 1.0 los electrodos que cuentan como pared y en 0.0 los excluidos —fuente, tubo, detector— antes de consultar cada punto), eliminando la dependencia del STL desalineado. Validada contra los 5 casos canónicos (candidatos con transmisión real conocida y candidatos muertos conocidos), la geometría nueva predijo 388 de 395 partículas limpias para la mejor solución conocida, frente a 368 de 384 medidas en SIMION real; la geometría anterior predecía cero partículas llegando al detector y 85% de choques contra pared fantasma para ese mismo candidato [C10]. El margen de contacto también se recalibró de $1.5 	ext{ mm}$ (el colchón que compensaba el desalineamiento del STL) a $0.5 	ext{ mm}$ (el margen físicamente correcto para paredes reales).

### 6.7 TPE como Motor Principal y la Integración de Semillas del GP
La mayor parte del desarrollo del proyecto se realizó con TPE (*Tree-structured Parzen Estimator*, el *sampler* por defecto de Optuna) combinado con el filtrado RK4; esta combinación, robusta y económica, produjo las primeras soluciones reales del gemelo. El GP (proceso gaussiano) ofrece una ventaja teórica que TPE no tiene —modela correlaciones entre variables mediante su kernel de covarianza, mientras que TPE estima cada parámetro de forma independiente—, pero consultar su modelo real (`ask() / sample_relative()`) cuesta $\sim 100	ext{–}300 	ext{ ms}$ y este costo crece con el historial, lo que resulta demasiado alto para generar los cientos de candidatos que RK4 filtra por iteración.

La integración adoptada paga el costo real del GP solo unas pocas veces por iteración ($n_{	ext{gp\_seeds}}$, típicamente 5): estas semillas informadas se mezclan dentro del mismo lote de candidatos que evalúa RK4, junto con perturbaciones económicas alrededor de ellas, de modo que se explora el espacio amplio con el mecanismo económico (TPE/RK4) mientras se explota el mejor punto identificado por el GP. El gráfico de convergencia muestra el resultado: durante las primeras $\sim 70	ext{–}80$ evaluaciones ambos motores avanzan de forma comparable, incluso con TPE adelante durante buena parte del tramo; recién cerca de la evaluación #80 el estudio *GP-seeded* encuentra un mínimo considerablemente mejor y lo sostiene, mientras TPE se estanca en un valor peor.

> **Figura 4.** Convergencia comparada del costo $J_{v2}$ durante la campaña TES-GP-GP6D: TPE (*gemelo_v2*), *GP-seeded* (*gemelo_db_v2*) y *bender6d*. El *GP-seeded* se despega recién cerca de la evaluación #80: su ventaja depende del presupuesto adicional de evaluaciones, no de una superioridad por evaluación individual.

### 6.8 Cuencas de Optimización: Evidencia de Múltiples Óptimos de Polaridad Opuesta
El espacio de soluciones que producen deflexión útil del haz es fino y sensible ($\sim 1	imes 10^{-9}$ del volumen total, [C2]), difícil de alcanzar por muestreo aleatorio. Al comparar los mejores candidatos de dos corridas *GP-seeded* independientes se identificaron dos regiones de solución con transmisión real, pero con la polaridad de varios electrodos invertida entre sí: una distancia de $1241 	ext{ V}$ (norma L2 sobre los 8 electrodos optimizables) entre ambos mejores candidatos, con signo opuesto en $V_3$, $V_6$ y $V_9$ [C11]. La corrida que quedó en la cuenca de menor transmisión no escapó de ella durante el total de sus evaluaciones, pese a usar el mismo mecanismo de búsqueda que la otra corrida: la diferencia fue el punto de arranque, no el algoritmo.

Esto se conecta directamente con la sección 6.4: ni TPE ni el GP, tal como están planteados, conocen la estructura física del problema (el patrón de pares diagonales correlacionados del cuadrupolo), por lo que ninguno cuenta con un mecanismo para evaluar, o saltar hacia, una cuenca mejor fuera de su vecindario de arranque.

### 6.9 Reparametrización del Bender: Alineación con la Estructura Física del Cuadrupolo
La evidencia de la sección 6.8 dejaba una limitación clara: el optimizador seguía explorando cuatro voltajes independientes cuando los datos mostraban que la solución se ubica sobre un patrón correlacionado. Para resolverla, se consolidaron los cuatro electrodos del doblador en tres variables de control físico: **A** como offset común, **B** como intensidad de flexión sobre el patrón diagonal ya derivado en el análisis de campo del *bender*, y **C** como asimetría dipolar horizontal, con el mapeo:

$$V_9 = A + B + C$$
$$V_{10} = A - B$$
$$V_{11} = A - B$$
$$V_{12} = A + B - C$$ [C12]

Con esto, la búsqueda dejó de operar sobre un espacio de 8 voltajes independientes y pasó a moverse sobre las coordenadas en las que el *hardware* realmente actúa.

Las cotas de estas variables se manejaron con el mismo criterio de la sección 6.5: dejar que los datos determinen el ajuste. Cuando el mejor valor de B se acercaba al borde sin mostrar una meseta, se ampliaron los rangos de A y B (de $\pm 500$ a $\pm 600 / \pm 800$); la misma señal aparece ahora en C —4 de los 5 mejores *hitters* de una corrida quedaron en su cota de $\pm 600$ [C12]—, por lo que la ampliación correspondiente queda pendiente como siguiente paso.

El resultado justificó el cambio: el récord absoluto del proyecto surgió de esta parametrización (113 de 500 iones con el *bender6d*; en una comparación directa a igual presupuesto, la versión reducida duplicó la media de *hits* del 8D original: 24.3 frente a 10.0 [C12]). De igual relevancia, la versión 7D encontró la cuenca asimétrica arrancando desde $0.0 	ext{ V}$, sin ninguna semilla sesgada: el tipo de descubrimiento orgánico que en la sección 6.8 se consideraba poco probable sin dotar a la búsqueda de la estructura física del problema.

### 6.10 Segunda Revisión de la Función Objetivo: Cambio de Régimen en el Éxito de la Búsqueda
Corregir el objetivo una vez no garantizó que no debiera corregirse de nuevo. El término de transmisión saturante que resolvió el problema del régimen de 0 *hits* —una solución correcta en ese momento— se convirtió en un problema cuando la búsqueda maduró: en el régimen actual de 70–110 *hits*, un *hit* marginal aporta apenas $\sim 0.0005$ al valor de $J$, y ninguna recalibración de esa forma funcional corrige la limitación [C13].

La consecuencia se observó directamente en los datos: el objetivo rankeaba una cuenca de 84 *hits* por encima de una de 105, porque los términos cosméticos del haz (*offset*, Twiss, colimación, halo) pesaban más que una diferencia de 21 *hits* reales. El optimizador no estaba fallando: estaba optimizando correctamente un criterio incorrecto.

La corrección mantiene el término saturante —que sigue aportando gradiente en el arranque de la búsqueda— y le suma un término lineal de transmisión que no satura, re-normalizando el total para conservar la escala $[0,1]$ del objetivo. Verificado contra los trials reales del proyecto, el nuevo objetivo reordena correctamente $105 > 95 > 84 > 74 	ext{ hits}$, con los términos cosméticos actuando como desempate entre configuraciones equivalentes en *hits* [C13]. La revisión también permitió identificar dos problemas colaterales: el puntaje de penalización por vuelo fallido había quedado por debajo del nuevo peor caso válido (una corrida de SIMION fallida habría rankeado mejor que una configuración real de 0 *hits*), y el test unitario del haz perfecto estaba desactualizado. Ambos se corrigieron, con un test de regresión nuevo que codifica este caso para prevenir su recurrencia. La conclusión general es que la función objetivo no es un artefacto que se calibra una única vez: cada cambio de régimen en la búsqueda exige revisarla nuevamente.

> **Figura 5.** Reordenamiento de cinco configuraciones reales del proyecto al pasar del objetivo v2.3 (término de transmisión saturante) al v2.4 (término lineal, $\lambda = 2$, re-normalizado). Con v2.3, el candidato de 84 *hits* rankeaba primero y el de 105 *hits*, último; con v2.4 el ranking sigue a los *hits* [C13].

### 6.11 Evaluación Formal y Selección del Gemelo Central
Para resolver la pregunta de cuál gemelo es mejor de forma objetiva, se construyó una rúbrica ejecutable que recorre todos los estudios persistidos y califica cada gemelo en siete dimensiones —admisibilidad física, dirección (*hits* del candidato recomendado), exactitud de ranking en la región informativa, linealidad de la consigna inversa, honestidad de la incertidumbre, alineación de gradiente contra SIMION y eficiencia de datos—, combinadas en un puntaje único [C14].

El resultado del ranking resume el estado del proyecto: el primer puesto (*bender6d*, 113 *hits*, puntaje 0.712) obtiene el mejor rendimiento crudo, pero presenta la peor exactitud de ranking fino de la tabla; el gemelo mejor calibrado (puntaje 0.631) encontró un óptimo menos extremo. Ningún gemelo domina simultáneamente en rendimiento pico y calibración fina.

Como gemelo central del proyecto se seleccionó **bender7d**, del estudio Search v2.1 (puntaje 0.573, tercer lugar del ranking; mejor $J$ de la parametrización 7D, con 74 *hits* en su configuración recomendada y 83 en su mejor *hitter*) [C14]. La decisión pondera que esta es la parametrización física completa (incluye la asimetría dipolar C), con una consigna inversa fuerte ($I = 0.852$) y la mayor honestidad de incertidumbre del conjunto —reconociendo que no posee el récord de *hits*, y que su eficiencia de datos es su punto más débil—. Todas las figuras del gemelo central se regeneraron a partir de su base de datos congelada en `studies/`.

> **Figura 6.** Rúbrica de evaluación formal aplicada a todos los estudios persistidos (top 5, contribuciones ponderadas por término). El gemelo central seleccionado (*bender7d*, Search v2.1) obtiene 0.573 [C14].

> **Figura 7.** Configuración de voltajes del gemelo central (mejor $J$ del estudio Search v2.1, trial #83, 74 *hits*): $V_3 = +310$, $V_6 = -372$, *bender* $A = -282$ / $B = -284$ / $C = +300$ ($V_9..V_{12}$ expandidos), $V_{15} = -67$, $V_{18} = -81$.

---

## 7. Reflexión

### 7.1 Confianza en el Gemelo Digital
Con el objetivo corregido (sección 6.3) y la geometría extraída del archivo PA (sección 6.6) integrados, el lazo autónomo (Optuna *GP-seeded* + filtro RK4 + SIMION real, sin intervención manual de voltajes) encontró un candidato que transmite 133 de 500 iones ($\sim 26.6\%$); una corrida independiente para verificar reproducibilidad dio 108 de 500 ($\sim 21.6\%$), diferencia atribuible al ruido de disparo ya documentado en [C6] y no a una regresión del modelo.

Las campañas con la parametrización reducida (sección 6.9) movieron el récord una vez más: 113 de 500 iones ($\sim 22.6\%$) con el *bender6d*, también de forma completamente autónoma [C12]; la variación entre corridas idénticas de esa misma configuración (113, 57 y 88 *hits*) confirma que el punto de arranque influye tanto como el método, consistente con [C6]. El resultado no corresponde al mínimo global del espacio de respuestas: las cuencas descritas en la sección 6.8 muestran que existe al menos una región alternativa de voltajes con transmisión real que el lazo no visitó en esta corrida.

Aun así, existen razones sólidas para confiar en el flujo propuesto y continuar su desarrollo:
1. **Rompe el Bloqueo de Cero Hits:** El flujo logra encontrar mínimos locales dentro de un espacio de soluciones que es, en su mayoría, cero —un objetivo que el flujo original, al realizar una búsqueda prácticamente aleatoria, no alcanzaba—: 26 o más evaluaciones aleatorias dieron cero *hits*, mientras que el flujo con conocimiento físico encontró su primer *hit* dentro de las primeras 3 a 7 evaluaciones de cada estudio iniciado desde cero, acumulando 86 trials con *hits* en total [C7][C8].
2. **Concordancia de Clasificación Gruesa:** Existe concordancia entre las simulaciones de RK4 y SIMION: el RK4 cumple su función principal al identificar candidatos que quedan fuera del espacio deseado, permitiendo descartar de antemano una gran cantidad de candidatos que producirían malos resultados en SIMION. Si bien un buen resultado en RK4 no garantiza un buen resultado en SIMION, lo contrario sí se cumple de forma consistente: en la validación sobre 288 configuraciones con verdad de terreno, el filtro concentró 17 de los 61 *hitters* reales en su *top-30* (el azar esperaría $\sim 6$, un enriquecimiento de $2.7	imes$ [C3]), y todos los *hits* reales del proyecto surgieron de candidatos que el filtro había ubicado en su *top-10*. El simulador RK4 identifica de forma confiable los vecindarios donde existe una solución.
3. **Eficiencia Computacional:** La implementación cumple su objetivo de eficiencia computacional: el simulador RK4 evalúa 239 candidatos en un tiempo similar al que SIMION requiere para evaluar 10 ($\sim 5.77 	ext{ s}$ por simulación en SIMION —5.04 s de preparación del arreglo más 0.73 s de vuelo— frente a $\sim 0.25 	ext{ s}$ por candidato en RK4, una relación de $\sim 23	imes$) [C1].
4. **Consistencia en los Límites:** El comportamiento de ambos simuladores en regiones fuera del setup experimental es equivalente: para voltajes aleatorios lejos de la cuenca de solución, el RK4 predice que todo el haz muere lejos del detector ($\sim 325 	ext{ mm}$ en promedio) y SIMION confirma un resultado equivalente ($\sim 324	ext{–}338 	ext{ mm}$); con los voltajes optimizables en cero, ambos simuladores predicen que el haz choca contra la pared en $z \sim 80 	ext{ mm}$ [C5]. Esto confirma que el simulador RK4 representa adecuadamente el comportamiento físico capturado por SIMION.

> **Figura 8.** Forma real del haz en SIMION para la mejor configuración del gemelo central: distribución de impactos (*splats*) en $z$ y en el plano XY, en la ventana del detector.

### 7.2 Limitaciones del Gemelo Digital
Así como el gemelo demostró ser confiable para orientar la búsqueda, los datos también señalan cuatro aspectos en los que su confiabilidad es limitada:

1. **El ranking fino dentro de la cuenca de solución:** Medido nuevamente tras la corrección del objetivo y la geometría del PA sobre la región informativa actual (candidatos con señal real, $n=47$): Pearson $+0.540$, Spearman $+0.267$ entre la predicción del gemelo y el resultado real de SIMION. Esta métrica es distinta del enriquecimiento del filtro RK4 reportado sobre el total de 288 candidatos archivados —cuántos *hitters* reales caen en su *top-30*, una tarea de clasificación gruesa—; aquí se trata de la correlación fina dentro de la región que ya se sabe tiene señal, una tarea de regresión considerablemente más difícil. Ambas mediciones son consistentes con la misma conclusión: el gemelo distingue con buena confiabilidad el vecindario bueno del malo, pero no ordena con precisión los candidatos dentro del vecindario bueno.
2. **La transmisión absoluta:** Con la geometría antigua (STL), el RK4 sin terminación de partículas predecía que $\sim 50\%$ del haz sobrevivía, donde SIMION registraba 1–2%. Con la geometría extraída del archivo PA (sección 6.6), esta sobreestimación se redujo pero no desapareció: el gemelo continúa prediciendo más transmisión de la que SIMION confirma para el mismo candidato, aunque ahora ordena correctamente configuraciones viables frente a no viables, donde antes invertía el orden (sección 6.6 y [C10]). En síntesis: el gemelo no cuantifica *hits* con precisión absoluta, pero sí ordena vecindarios correctamente, y así es como debe utilizarse.
3. **Geometría de colisiones inicialmente aproximada (Resuelto):** Esta limitación ya fue resuelta (ver sección 6.6). La transformación de los archivos STL al marco de referencia del experimento cargaba un residuo medido de $\sim 25 	ext{ mm}$ en el cuadrupolo, causa directa de que la terminación de partículas fallara —el gemelo eliminaba partículas en ubicaciones incorrectas, perdiendo la información de vuelo que sí discrimina—. Esto se corrigió extrayendo la geometría directamente del archivo de potenciales de SIMION (sección 6.6); el margen de colisión se redujo de $1.5 	ext{ mm}$ a $0.5 	ext{ mm}$. Se conserva este punto en la lista de limitaciones, marcado como resuelto, para mantener trazable el antes y el después.
4. **La propia verdad de terreno es ruidosa:** El mismo conjunto de voltajes produjo 2, 4, 5, 8 y 10 *hits* en corridas repetidas e idénticas de SIMION, debido a que el haz se regenera aleatoriamente en cada vuelo (sin semilla fija) [C6]. A transmisiones de $\sim 2\%$, el conteo de *hits* tiene un ruido comparable a la señal misma; por esta razón, el objetivo denso del flujo utiliza los 500 puntos de impacto de cada corrida, y no solo los $\sim 10$ que entran en la ventana del detector.

> **Figura 9.** Validación del gemelo central en la región informativa: predicción de costo del RK4 frente al costo real medido en SIMION, coloreado por número de *hits*. El gemelo separa el vecindario bueno del malo, pero no ordena con precisión fina dentro del vecindario bueno.

---

### Referencias de Cálculo
Derivación completa de cada número citado: `docs/CALCULOS_INFORME.txt`. Bases de datos de estudios: `studies/` (el gemelo central se encuentra en `studies/gemelo_v2_bender7d.db`). Registro de corridas: `studies/registro_corridas.jsonl`.

* **[C1]** Costo por candidato: SIMION vs. RK4 (relación $\sim 23	imes$)
* **[C2]** Datos necesarios para un modelo estocástico (*surrogate*)
* **[C3]** Enriquecimiento del filtro RK4 ($2.7	imes$ sobre el azar)
* **[C4]** Escalado del costo del gemelo con la complejidad del arreglo
* **[C5]** Concordancia fuera de la región experimental
* **[C6]** Ruido de la verdad de terreno (SIMION)
* **[C7]** Presupuesto total del proyecto
* **[C8]** Eficiencia de búsqueda con punto inicial físico
* **[C9]** Bug del objetivo $J_{v2}$: transmisión lineal vs. racional
* **[C10]** Geometría de colisiones: de los STL al archivo `.PA0`
* **[C11]** Cuencas de optimización: dos óptimos de polaridad opuesta
* **[C12]** Reparametrización del bender: coordenadas A/B/C y gestión de cotas
* **[C13]** Segunda falla del objetivo: saturación en el régimen alto de hits
* **[C14]** Rúbrica de evaluación formal y selección del gemelo central
