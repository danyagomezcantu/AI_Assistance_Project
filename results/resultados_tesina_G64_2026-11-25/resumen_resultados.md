# Resultados de la tesina, C0 a 64³

Generado el 2026-11-25. Cinco pliegues por paciente (semilla 42); cada paciente aparece una sola vez en prueba. Distancias en mm del paciente contra el STL original.

Cohorte: 223 de 367 pares; excluidos 144 (molde abierto: 104, no pasa A1: 100) con la regla de P3b, fijada antes de ver resultados.

## Mediana [RIC] agrupada y media ± DE entre pliegues

| Métrica | Método | n | Mediana [Q1, Q3] | Media ± DE pliegues |
|---|---|---|---|---|
| dice | B1 | 223 | 0.749 [0.670, 0.797] | 0.705 ± 0.019 |
| dice | B2 | 223 | 0.770 [0.698, 0.825] | 0.740 ± 0.016 |
| dice | C0 | 223 | 0.865 [0.817, 0.894] | 0.846 ± 0.008 |
| iou | B1 | 223 | 0.599 [0.504, 0.663] | 0.561 ± 0.020 |
| iou | B2 | 223 | 0.626 [0.537, 0.702] | 0.600 ± 0.018 |
| iou | C0 | 223 | 0.762 [0.690, 0.809] | 0.740 ± 0.010 |
| chamfer_total | B1 | 223 | 5.895 [4.785, 7.632] | 7.549 ± 0.684 |
| chamfer_total | B2 | 223 | 5.218 [3.682, 6.717] | 6.125 ± 0.575 |
| chamfer_total | C0 | 223 | 3.243 [2.452, 4.213] | 3.583 ± 0.159 |
| chamfer_frente | B1 | 223 | 5.213 [4.284, 6.521] | 6.218 ± 0.716 |
| chamfer_frente | B2 | 223 | 3.693 [3.094, 4.457] | 4.396 ± 0.394 |
| chamfer_frente | C0 | 223 | 1.808 [1.583, 2.122] | 2.068 ± 0.158 |
| chamfer_base | B1 | 223 | 6.235 [4.743, 9.054] | 8.233 ± 0.668 |
| chamfer_base | B2 | 223 | 6.396 [4.098, 9.004] | 7.451 ± 0.710 |
| chamfer_base | C0 | 223 | 4.502 [3.066, 5.919] | 4.870 ± 0.248 |
| hd95_total | B1 | 223 | 14.904 [12.172, 19.681] | 19.994 ± 2.353 |
| hd95_total | B2 | 223 | 13.343 [9.915, 18.112] | 16.965 ± 1.841 |
| hd95_total | C0 | 223 | 9.507 [7.185, 12.506] | 10.930 ± 0.683 |
| hd95_frente | B1 | 223 | 12.868 [11.479, 16.205] | 16.593 ± 2.135 |
| hd95_frente | B2 | 223 | 10.034 [7.932, 11.815] | 12.416 ± 1.455 |
| hd95_frente | C0 | 223 | 5.972 [5.109, 7.180] | 7.204 ± 0.638 |
| hd95_base | B1 | 223 | 15.309 [12.064, 20.956] | 20.416 ± 2.315 |
| hd95_base | B2 | 223 | 14.203 [10.707, 19.522] | 17.847 ± 1.896 |
| hd95_base | C0 | 223 | 10.397 [7.788, 13.821] | 11.847 ± 0.720 |

## Comparaciones pareadas (Wilcoxon, Holm, α = 0.05)

| Comparación | Métrica | n | Mediana dif. | IC95 % | r | p | p Holm | Favorece |
|---|---|---|---|---|---|---|---|---|
| C0 vs B1 | dice | 223 | 0.118 | [0.103, 0.125] | 0.85 | 0.0000 | 0.0000 | C0 * |
| C0 vs B1 | iou | 223 | 0.162 | [0.143, 0.177] | 0.86 | 0.0000 | 0.0000 | C0 * |
| C0 vs B1 | chamfer_total | 223 | -2.687 | [-2.967, -2.413] | -0.86 | 0.0000 | 0.0000 | C0 * |
| C0 vs B1 | chamfer_frente | 223 | -3.307 | [-3.523, -3.077] | -0.87 | 0.0000 | 0.0000 | C0 * |
| C0 vs B1 | chamfer_base | 223 | -2.180 | [-2.484, -1.852] | -0.73 | 0.0000 | 0.0000 | C0 * |
| C0 vs B1 | hd95_total | 223 | -5.603 | [-6.458, -4.907] | -0.81 | 0.0000 | 0.0000 | C0 * |
| C0 vs B1 | hd95_frente | 223 | -6.999 | [-7.564, -6.576] | -0.86 | 0.0000 | 0.0000 | C0 * |
| C0 vs B1 | hd95_base | 223 | -5.364 | [-6.017, -4.475] | -0.76 | 0.0000 | 0.0000 | C0 * |
| C0 vs B2 | dice | 223 | 0.095 | [0.086, 0.106] | 0.78 | 0.0000 | 0.0000 | C0 * |
| C0 vs B2 | iou | 223 | 0.136 | [0.123, 0.144] | 0.78 | 0.0000 | 0.0000 | C0 * |
| C0 vs B2 | chamfer_total | 223 | -1.922 | [-2.188, -1.624] | -0.77 | 0.0000 | 0.0000 | C0 * |
| C0 vs B2 | chamfer_frente | 223 | -1.770 | [-1.919, -1.634] | -0.86 | 0.0000 | 0.0000 | C0 * |
| C0 vs B2 | chamfer_base | 223 | -1.836 | [-2.279, -1.409] | -0.60 | 0.0000 | 0.0000 | C0 * |
| C0 vs B2 | hd95_total | 223 | -3.923 | [-5.002, -3.184] | -0.65 | 0.0000 | 0.0000 | C0 * |
| C0 vs B2 | hd95_frente | 223 | -3.396 | [-3.926, -3.102] | -0.81 | 0.0000 | 0.0000 | C0 * |
| C0 vs B2 | hd95_base | 223 | -4.153 | [-5.357, -3.445] | -0.61 | 0.0000 | 0.0000 | C0 * |

## Costo de cómputo

Entrenamiento por pliegue: mediana 3.4 min; memoria pico de GPU: máximo 314 MB; GPU: Tesla T4; parámetros: 1,401,265.
Posprocesamiento (marching cubes, suavizado, mm, métricas), mediana por paciente: B1 1.92 s, B2 0.45 s, C0 2.72 s.

## Viabilidad técnica

Entrenamiento de C0: 0.31 h de GPU en 5 pliegues; inferencia de la red 0.006 s por paciente.
Preprocesamiento: mediana 14.9 s por paciente (1.60 h en total, incluye el registro).
Prototipo integrado (rostro STL → molde y plantilla STL): mediana 2.1 s por paciente, 0.8 s sin contar la carga del modelo.

Archivos de este paquete: cajas_metricas.png, config.json, costo_computo.csv, entorno.json, exclusiones_resumen.json, indice_conteos.json, metricas_por_paciente_anon.csv, plantillas_resumen.csv, preprocesamiento_resumen.json, prototipo_tiempos.csv, pruebas_estadisticas.csv, pruebas_suavizado.csv, resumen_por_metodo.csv, suavizado_resumen.csv, verificaciones_resumen.csv, viabilidad_tecnica.json.
