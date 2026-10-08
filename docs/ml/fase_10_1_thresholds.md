# Fase 10.1 — Evaluación avanzada de thresholds

## Objetivo

Analizar cómo cambia el desempeño de dos modelos de regresión logística al modificar el umbral de clasificación, sin realizar nuevos entrenamientos.

## Datos de evaluación

- Dataset: `validation`
- Registros: 61,503
- Variables: 378
- Positivos: 4,965
- Negativos: 56,538

## Metodología

Se reutilizaron los modelos entrenados y sus respectivos escaladores MaxAbsScaler. Para cada threshold se calcularon precision, recall, F1-score y matriz de confusión.

Se verificó que ambos modelos reprodujeran sus matrices de confusión originales con threshold 0.50.

## Resultados

### Modelo: unbalanced

| Threshold | Precision | Recall | F1 | TP | FP | FN | Alertas |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 0.0839 | 0.9962 | 0.1548 | 4946 | 53994 | 19 | 58940 |
| 0.02 | 0.0939 | 0.9708 | 0.1712 | 4820 | 46511 | 145 | 51331 |
| 0.03 | 0.1070 | 0.9378 | 0.1920 | 4656 | 38870 | 309 | 43526 |
| 0.04 | 0.1198 | 0.8884 | 0.2111 | 4411 | 32422 | 554 | 36833 |
| 0.05 | 0.1340 | 0.8431 | 0.2312 | 4186 | 27053 | 779 | 31239 |
| 0.10 | 0.1977 | 0.6216 | 0.2999 | 3086 | 12527 | 1879 | 15613 |
| 0.15 | 0.2518 | 0.4423 | 0.3209 | 2196 | 6526 | 2769 | 8722 |
| 0.20 | 0.3016 | 0.3132 | 0.3073 | 1555 | 3601 | 3410 | 5156 |
| 0.25 | 0.3522 | 0.2234 | 0.2734 | 1109 | 2040 | 3856 | 3149 |
| 0.30 | 0.3913 | 0.1567 | 0.2238 | 778 | 1210 | 4187 | 1988 |
| 0.35 | 0.4385 | 0.1078 | 0.1730 | 535 | 685 | 4430 | 1220 |
| 0.40 | 0.4707 | 0.0729 | 0.1263 | 362 | 407 | 4603 | 769 |
| 0.45 | 0.5352 | 0.0489 | 0.0897 | 243 | 211 | 4722 | 454 |
| 0.50 | 0.5762 | 0.0312 | 0.0592 | 155 | 114 | 4810 | 269 |
| 0.55 | 0.6154 | 0.0193 | 0.0375 | 96 | 60 | 4869 | 156 |
| 0.60 | 0.7375 | 0.0119 | 0.0234 | 59 | 21 | 4906 | 80 |
| 0.65 | 0.8000 | 0.0064 | 0.0128 | 32 | 8 | 4933 | 40 |
| 0.70 | 0.7826 | 0.0036 | 0.0072 | 18 | 5 | 4947 | 23 |
| 0.75 | 0.8000 | 0.0008 | 0.0016 | 4 | 1 | 4961 | 5 |
| 0.80 | 1.0000 | 0.0006 | 0.0012 | 3 | 0 | 4962 | 3 |
| 0.85 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 4965 | 0 |
| 0.90 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 4965 | 0 |
| 0.95 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 4965 | 0 |

**Mejor F1 evaluado:** 0.3209 con threshold 0.15.

### Modelo: balanced

| Threshold | Precision | Recall | F1 | TP | FP | FN | Alertas |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 0.0807 | 1.0000 | 0.1494 | 4965 | 56535 | 0 | 61500 |
| 0.02 | 0.0808 | 1.0000 | 0.1495 | 4965 | 56501 | 0 | 61466 |
| 0.03 | 0.0809 | 1.0000 | 0.1496 | 4965 | 56435 | 0 | 61400 |
| 0.04 | 0.0810 | 0.9998 | 0.1499 | 4964 | 56305 | 1 | 61269 |
| 0.05 | 0.0813 | 0.9996 | 0.1504 | 4963 | 56088 | 2 | 61051 |
| 0.10 | 0.0843 | 0.9962 | 0.1555 | 4946 | 53720 | 19 | 58666 |
| 0.15 | 0.0896 | 0.9821 | 0.1642 | 4876 | 49561 | 89 | 54437 |
| 0.20 | 0.0971 | 0.9644 | 0.1765 | 4788 | 44501 | 177 | 49289 |
| 0.25 | 0.1067 | 0.9398 | 0.1916 | 4666 | 39081 | 299 | 43747 |
| 0.30 | 0.1173 | 0.9053 | 0.2077 | 4495 | 33830 | 470 | 38325 |
| 0.35 | 0.1288 | 0.8596 | 0.2240 | 4268 | 28875 | 697 | 33143 |
| 0.40 | 0.1423 | 0.8109 | 0.2422 | 4026 | 24257 | 939 | 28283 |
| 0.45 | 0.1578 | 0.7603 | 0.2614 | 3775 | 20148 | 1190 | 23923 |
| 0.50 | 0.1741 | 0.6983 | 0.2787 | 3467 | 16444 | 1498 | 19911 |
| 0.55 | 0.1935 | 0.6314 | 0.2962 | 3135 | 13066 | 1830 | 16201 |
| 0.60 | 0.2155 | 0.5535 | 0.3102 | 2748 | 10003 | 2217 | 12751 |
| 0.65 | 0.2397 | 0.4733 | 0.3182 | 2350 | 7455 | 2615 | 9805 |
| 0.70 | 0.2701 | 0.3911 | 0.3196 | 1942 | 5247 | 3023 | 7189 |
| 0.75 | 0.3060 | 0.3041 | 0.3051 | 1510 | 3424 | 3455 | 4934 |
| 0.80 | 0.3539 | 0.2179 | 0.2698 | 1082 | 1975 | 3883 | 3057 |
| 0.85 | 0.4033 | 0.1343 | 0.2015 | 667 | 987 | 4298 | 1654 |
| 0.90 | 0.4975 | 0.0592 | 0.1058 | 294 | 297 | 4671 | 591 |
| 0.95 | 0.6707 | 0.0111 | 0.0218 | 55 | 27 | 4910 | 82 |

**Mejor F1 evaluado:** 0.3196 con threshold 0.70.

## Interpretación

El threshold 0.50 no necesariamente proporciona el mejor equilibrio entre precision y recall. Los modelos pueden necesitar thresholds diferentes porque producen distribuciones de probabilidades distintas.

## Limitaciones

- Los resultados corresponden al conjunto de validación.
- Se evaluó una lista discreta de thresholds.
- Maximizar F1 no equivale a optimizar el beneficio financiero.
- TARGET=1 representa dificultades de pago, no recuperación monetaria de cartera castigada.
- Se requiere una evaluación independiente antes de considerar una implementación productiva.

## Artefacto

`data/artifacts/ml/advanced_threshold_evaluation.json`
