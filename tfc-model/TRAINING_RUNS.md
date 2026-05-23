# Registro de ejecuciones

Este archivo registra tiempos de ejecucion y consumo de recursos para comparar el proyecto entre entornos como local, Colab e IsardVDI.

No incluye comparaciones historicas de calidad entre versiones de modelos.

## 2026-05-22 - Consumo XGBoost, ejecucion parcial

| Campo | Valor |
|---|---:|
| Estado | completado |
| Entorno | Windows local |
| Python | 3.12.0 |
| Sectores entrenados | 4 |
| RandomizedSearchCV total | 770.9 s |
| total | 180.1 s |
| residencial | 274.2 s |
| servicios | 180.4 s |
| industria | 136.2 s |
| Recursos | no medido |

## 2026-05-22 - Consumo XGBoost, ejecucion completa

| Campo | Valor |
|---|---:|
| Estado | completado |
| Entorno | Windows local |
| Python | 3.12.0 |
| Sectores entrenados | 4 |
| RandomizedSearchCV total | 536.5 s |
| total | 136.6 s |
| residencial | 127.0 s |
| servicios | 140.8 s |
| industria | 132.1 s |
| Recursos | no medido |

## 2026-05-22 - Eolica HGB vs Ridge

| Campo | Valor |
|---|---:|
| Estado | completado |
| Entorno | Windows local |
| Python | 3.12.0 |
| Modelos entrenados | 2 |
| Ridge RandomizedSearchCV | 7.3 s |
| HGB RandomizedSearchCV | 11.9 s |
| Tiempo total busquedas | 19.2 s |
| Recursos | no medido |

## 2026-05-22 - Benchmark API

| Campo | Valor |
|---|---:|
| Estado | completado |
| Entorno | Windows local |
| Plataforma | Windows-11-10.0.26200-SP0 |
| Python | 3.12.0 |
| CPU logicas disponibles | 16 |
| Memoria RSS proceso | 318.0 MB |
| Pico tracemalloc | 0.87 MB |
| Tiempo total benchmark | 1811.58 ms |
| predict_consumption media | 33.02 ms |
| predict_consumption min | 30.19 ms |
| predict_consumption max | 34.48 ms |
| predict_eolica media | 298.69 ms |
| predict_eolica min | 210.81 ms |
| predict_eolica max | 621.13 ms |

## 2026-05-23 - Feature selection robusta

| Campo | Valor |
|---|---:|
| Estado | completado |
| Entorno | Windows local |
| Python | 3.12.0 |
| Script | `src/training/robust_feature_selection_experiment.py --problem all` |
| Problemas evaluados | consumo + eolica |
| Selectores | full_current, robust_physical, correlation_filter, model_importance, mutual_info, permutation_importance |
| Folds temporales | 4 |
| Tamaños reducidos | 10, 15, 20 |
| Tiempo total | 719.7 s |
| Recursos | no medido |

## 2026-05-23 - Open-Meteo ampliacion territorial

| Campo | Valor |
|---|---:|
| Estado | completado |
| Entorno | Windows local |
| Python | 3.12.0 |
| Script | `tfc-datasets/scripts/build_open_meteo_augmented_dataset.py --start 2020-01-01 --end 2025-06-30` |
| Municipios ISTAC consumo | 87 |
| Municipios consumo antes | 39 |
| Municipios Open-Meteo añadidos | 48 |
| Municipios consumo despues | 87 |
| Filas Open-Meteo descargadas | 96,384 |
| Filas dataset consumo ampliado | 165,106 |
| Periodo | 2020-01-01 -> 2025-06-30 |
| Tiempo total descarga + evaluacion | 1013.9 s |
| Recursos descarga + evaluacion | no medido |

## 2026-05-23 - Open-Meteo ampliacion territorial, ejecucion con cache

| Campo | Valor |
|---|---:|
| Estado | completado |
| Entorno | Windows local |
| Plataforma | Windows-11-10.0.26200-SP0 |
| Python | 3.12.0 |
| Script | `tfc-datasets/scripts/build_open_meteo_augmented_dataset.py --start 2020-01-01 --end 2025-06-30 --skip-fetch` |
| Municipios consumo despues | 87 |
| Filas Open-Meteo reutilizadas | 96,384 |
| Filas dataset consumo ampliado | 165,106 |
| Tiempo total reconstruccion + evaluacion | 58.18 s |
| CPU logicas disponibles | 16 |
| Memoria RSS proceso | 413.46 MB |
| Pico tracemalloc | 290.98 MB |

## 2026-05-23 - Reentrenamiento oficial consumo con dataset ampliado

| Campo | Valor |
|---|---:|
| Estado | completado |
| Entorno | Windows local |
| Script | `tfc-model/src/training/Energy_Consumption/XGBoost.py` |
| Dataset | `tfc-model/data/final_demand_consumption_dataset.csv` |
| Municipios | 87 |
| Filas dataset | 165,106 |
| Sectores entrenados | 4 |
| Tiempo total | 1016.86 s |
| Recursos | no medido |

## 2026-05-23 - Reentrenamiento oficial eolica con dataset ampliado

| Campo | Valor |
|---|---:|
| Estado | completado |
| Entorno | Windows local |
| Script | `tfc-model/src/training/Renewable_Energy_Generation/HistGradientBoosting_VS_Ridge.py` |
| Dataset | `tfc-model/data/final_renewable_generation_dataset.csv` |
| Filas dataset | 2,008 |
| Modelos entrenados | 2 |
| Tiempo total | 30.65 s |
| Recursos | no medido |
