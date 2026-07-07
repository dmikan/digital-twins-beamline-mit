# Comparacion TPE vs GP -- desde cero (loop, sin sembrados)

Ambos re-corridos desde el punto del bender-sweep. Solo trials ENCONTRADOS
POR EL LOOP (excluidos los sembrados del registro). NOTA: los .db fluctuan
por la sincronizacion de OneDrive; snapshot al momento de la consulta.

## Presupuesto completo (tal como quedaron)
| metrica | TPE (gemelo_v2) | GP (gemelo_db_v2) |
|---|---|---|
| evaluaciones del loop | 60 | 150 |
| **best hitter** | **49 hits** (eval #47) | **133 hits** (eval #80) |
| mejor J_v2 | 0.283 | 0.212 |
| top-5 hits | [49, 45, 43, 33, 31] | [133, 117, 80, 74, 68] |
| media top-5 | 40.2 | 94.4 |
| media hits (todos) | 7.6 | 18.3 |
| trials >=50 hits | 0 | 11 |

## Mismo presupuesto (60 evaluaciones cada uno) -- comparacion JUSTA
| metrica | TPE (60) | GP (primeras 60) |
|---|---|---|
| best hitter | 49 hits | 42 hits |
| mejor J_v2 | 0.283 | 0.281 |
| media hits | 7.6 | 10.0 |
| trials >=40 | 3 | 2 |

## Donde se encontro el best hitter (por el LOOP, sin sembrar)
- **TPE**: 49 hits en la evaluacion #47 de 60.
- **GP** : 133 hits en la evaluacion #80 de 150.

## Lectura (consistente con los datos)
- A **presupuesto completo** el GP gana amplio (133 vs 49 hits,
  media 18.3 vs 7.6) -- PERO corrio 150 evals vs 60 del TPE.
- A **MISMO presupuesto (60 evals)** la cosa se empareja: pico TPE
  (49 TPE vs 42 GP), J casi empatada (0.283 vs 0.281),
  GP con media algo mayor (10.0 vs 7.6).
- Conclusion honesta: **la ventaja titular del GP es principalmente PRESUPUESTO,
  no superioridad del sampler.** A igual numero de evaluaciones son comparables;
  el GP sigue pagando su exploracion tarde (el pico de 133 llego en la eval
  #80), asi que con mas budget se despega.
