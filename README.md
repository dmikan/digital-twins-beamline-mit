# 🔬 Gemelo Digital de una Línea de Haz de Iones

> Un gemelo digital basado en física para optimizar una línea de haz de iones con deflexión de 90°, usando Procesos Gaussianos, simulación RK4 y SIMION como referencia de alta fidelidad.

---

## Descripción general

Este proyecto implementa un **gemelo digital** de una línea de haz de iones electrostática que deflecta un haz de Si⁺ 90° desde la fuente hasta el detector. El sistema utiliza un lazo de optimización en dos etapas que combina un modelo sustituto rápido (RK4 basado en física) con simulaciones de alta fidelidad en SIMION, guiadas por optimización bayesiana (Procesos Gaussianos vía Optuna).

La línea de haz consta de **19 electrodos**. El gemelo digital optimiza **8 voltajes de electrodo libres** (electrodos 3, 6, 9–12, 15, 18; rango ±1000 V) para maximizar la transmisión de iones al detector, manteniendo el resto fijos (fuente: +500 V, detector: −2000 V, tierras).

**Parámetros del haz de iones:**
- Especie: Si⁺ (28 amu, 15 eV)
- Haz inicial: 500 iones en un cono de 15° desde la posición (395, 75, 77) mm
- Región del detector: x ∈ [70–82] mm, y ∈ [70–83] mm, z ∈ [403–407] mm

---

## Resultados clave

| Métrica | Valor |
|---|---|
| Velocidad de evaluación RK4 | ~0,25 s/candidato |
| Velocidad de evaluación SIMION | ~5,77 s/candidato |
| Relación de velocidad (RK4 vs SIMION) | **~23×** más rápido |
| Enriquecimiento del filtro RK4 sobre azar | **2,7×** |
| Fracción de la zona con hits en el espacio de búsqueda | ~1 × 10⁻⁹ |
| Mejor configuración conocida | Parametrización bender 7d (Search v2.1) |

---

## Arquitectura

```
Optimizador (Optuna TPE / GP)
        │
        ▼
 [Etapa A – Screening económico]
  Motor de física RK4
  239 candidatos × 50 partículas × 1500 pasos
        │  top candidatos
        ▼
 [Etapa B – Screening fino]
  Motor de física RK4
  top-K × 200 partículas × 3000 pasos
        │  top-10
        ▼
 [Evaluación de referencia]
  SIMION 8.1
        │  resultados retroalimentan
        └──────────────────────► Base de datos Optuna (studies/)
```

El **motor RK4** aprovecha la linealidad de la ecuación de Laplace: el campo eléctrico para cualquier configuración de voltajes se obtiene como superposición lineal de 19 campos base precalculados, cargados una sola vez al inicio. Las trayectorias se integran en un lote vectorizado (todos los candidatos y partículas simultáneamente) usando NumPy.

---

## Estructura del proyecto

```
.
├── gemelo_GP.py            # Gemelo digital entrenable con GP (punto de entrada ML)
├── caracterizador.py       # Caracterizador del haz: calcula el objetivo J_v2.4
├── physics.py              # Integrador RK4, grilla compuesta, colisión con paredes
├── optimizer.py            # Interfaz SIMION, definición de electrodos (base de cátedra)
├── basis_electrode_*.csv   # 19 mapas de campo eléctrico base precalculados
├── SimpleSetUp.*           # Archivos de configuración SIMION (.iob, .fly2, .rec, .lua)
├── derived_starting_point.json  # Punto de partida derivado por análisis físico
│
├── tools/                  # Scripts de análisis y utilidades
│   ├── fresh_run.py        # Corrida de benchmark limpia y reproducible
│   ├── validate_rk4_filter.py   # Validación offline del filtro RK4
│   ├── bender_field_analysis.py # Análisis de campo del cuadrupolo y punto inicial
│   ├── sim_batch.py        # Screening RK4 sin SIMION (exploración rápida)
│   ├── steer_scan.py       # Barrido de electrodos de dirección
│   ├── plot_rk4_trajectories.py # Visualización 3D de trayectorias
│   ├── report_figures.py   # Figuras de publicación
│   ├── generate_report_pdf.py   # Generador de reporte en PDF
│   └── especificacion_gemelo_digital.md  # Especificación técnica completa
│
├── studies/                # Bases de datos SQLite de Optuna (estudios activos)
│   ├── gemelo_db_v2.db
│   ├── gemelo_v2_bender6d.db
│   └── gemelo_v2_bender7d.db
│
├── outputs/                # Figuras, CSVs y reportes generados
├── docs/                   # Guías, cálculos y bitácora
│   ├── GUIA_CODIGO.txt     # Guía detallada del código
│   └── CALCULOS_INFORME.txt  # Cálculos de respaldo del informe
├── playpen/                # Área de trabajo libre (no usada por el gemelo)
└── legacy/                 # Estudios archivados y versiones superseded
```

---

## Inicio rápido

### Prerrequisitos

```bash
pip install numpy scipy optuna trimesh matplotlib
# Para el sampler GP (opcional):
# pip install torch
# export KMP_DUPLICATE_LIB_OK=TRUE
```

También se necesita:
- **SIMION 8.1** instalado (configurar `SIMION_INSTALL_DIR` en `optimizer.py`)
- `basis_electrode_1..19.csv` — mapas de campo base (incluidos)
- `SimpleSetUp.iob` y `electrode_.PA0` — archivos de configuración SIMION (incluidos)

### Uso de la fachada del gemelo digital

```python
from gemelo_GP import GemeloEntrenable

tw = GemeloEntrenable()

# Ver la mejor configuración actual
print(tw.resumen())
print(tw.mejor())

# Predecir con el modelo físico RK4 (~segundos)
tw.predecir(voltajes)

# Predecir con el modelo estadístico GP (instantáneo, con incertidumbre)
tw.predecir_modelo(voltajes)

# Evaluar con una corrida real de SIMION (~6 s)
tw.evaluar(voltajes)

# Entrenar el sustituto GP con el historial del estudio
tw.entrenar_modelo()

# Sugerir nuevos candidatos por Expected Improvement
for candidato in tw.sugerir(3):
    print(candidato)

# Ejecutar un ciclo completo de optimización (consume presupuesto SIMION)
tw.ciclo(rondas=1, k=2)
```

### Ejecutar el lazo de optimización

```bash
# Corrida de benchmark limpia — 50 evaluaciones SIMION, estudio nuevo
python tools/fresh_run.py 50

# Continuar un estudio existente
python tools/fresh_run.py 20 --continue

# Validar el filtro RK4 sin gastar SIMION
python tools/validate_rk4_filter.py

# Regenerar el punto de partida físico
python tools/bender_field_analysis.py
```

---

## La función objetivo: J_v2.4

El sistema minimiza un objetivo escalar $J_{v2.4} \in [0, 1]$ que combina múltiples métricas de calidad del haz en un único valor (menor = mejor):

$$J_{v2.4} = \text{normalizar}\left[\alpha \cdot d_{\text{top10\%}} + \lambda\!\left(1 - \frac{\text{hits}}{N_{\text{ref}}}\right) + \text{términos}(\sigma_x, \sigma_y, \text{halo}, \text{kurtosis}, \ldots)\right]$$

| Término | Descripción |
|---|---|
| $d_{\text{top10\%}}$ | Distancia media del 10% de iones más cercanos al detector |
| $\lambda = 2$ | Peso de transmisión lineal (evita saturación con muchos hits) |
| $\sigma_x, \sigma_y$ | Tamaño RMS del haz |
| fracción de halo | Fracción de partículas fuera del núcleo del haz |
| kurtosis | Agudeza de la distribución |
| divergencia angular | Ángulo de apertura del haz |
| emitancia de Twiss | Calidad del haz en el espacio de fases |

---

## Reparametrización del cuadrupolo

Para reducir la dimensionalidad efectiva del problema, los cuatro voltajes del cuadrupolo (V₉–V₁₂) se codifican con tres parámetros físicos:

| Parámetro | Significado físico |
|---|---|
| **A** | Desplazamiento común (offset) |
| **B** | Intensidad principal de flexión |
| **C** | Término de asimetría horizontal |

$$V_9 = A + B + C \quad V_{10} = A - B \quad V_{11} = A - B \quad V_{12} = A + B - C$$

Esta codificación preserva la simetría física del cuadrupolo y reduce la dimensionalidad del espacio de búsqueda.

---

## Motor de física RK4

El sustituto rápido integra las trayectorias de iones con un integrador Runge-Kutta de cuarto orden:

- **Paso temporal:** $dt = 10^{-8}$ s → ~0,6 mm por paso
- **Superposición de campos base:** campo = combinación lineal de 19 campos base precalculados
- **Grilla compuesta:** cajas locales de alta resolución (1,0 mm cuadrupolo, 2,5 mm colimadores) sobre una grilla global más gruesa (2,0 mm) → **reducción de memoria 15×**
- **Colisión con paredes:** vóxeles metálicos extraídos del archivo `.PA0` de SIMION, consultados via árbol espacial cada 3 pasos de integración (aceleración 4,2× respecto a chequear en cada paso)

---

## Notas de reproducibilidad

- Semilla del haz etapa A: **42** | Semilla etapa B: **1234**
- La fuente de iones de SIMION no tiene semilla fijada → el conteo de hits puede fluctuar levemente entre corridas idénticas
- `fresh_run.py` rechaza sobreescribir una base de datos existente, garantizando verdaderos arranques desde cero
- Las semillas de perturbación son deterministas: `semilla = 1000 × n_trials + iteración`

---

## Documentación

| Archivo | Contenido |
|---|---|
| [`docs/GUIA_CODIGO.txt`](docs/GUIA_CODIGO.txt) | Guía completa del código: qué hace cada archivo, cómo ejecutar cada flujo, parámetros ajustables |
| [`docs/CALCULOS_INFORME.txt`](docs/CALCULOS_INFORME.txt) | Cálculos de respaldo citados en el informe técnico |
| [`tools/especificacion_gemelo_digital.md`](tools/especificacion_gemelo_digital.md) | Especificación técnica completa: arquitectura, hiperparámetros, metodología de validación |

---

## Licencia

Este proyecto fue desarrollado como parte de una colaboración de investigación. Por favor contacta a los autores antes de reutilizar o redistribuir.
