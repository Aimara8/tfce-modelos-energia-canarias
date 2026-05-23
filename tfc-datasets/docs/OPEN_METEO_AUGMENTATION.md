# Open-Meteo territorial augmentation

Municipios actuales del modelo de consumo: **39**.
Municipios añadidos con Open-Meteo: **48**.
Total resultante: **87/87** municipios ISTAC.

Open-Meteo se usa solo para municipios sin meteo observada. Las estaciones reales mantienen prioridad.

Nota: la agregacion meteorologica para eolica puede mostrar 88 municipios porque incluye `Frontera`, que tiene estacion observada pero no aparece en el dataset ISTAC de consumo.

## Comparativa de modelos

| Problem | Scenario | Target | Municipios | MAE | RMSE | R2 | MAPE | WMAPE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| consumption | observed_current_39 | total | 39 | 8.8459 | 24.1265 | 0.9986 | 4.25 | 2.06 |
| consumption | observed_current_39 | residencial | 39 | 2.0715 | 5.1836 | 0.9995 | 2.58 | 1.44 |
| consumption | observed_current_39 | servicios | 39 | 6.7634 | 18.1408 | 0.9978 | 5.80 | 2.53 |
| consumption | observed_current_39 | industria | 39 | 2.0873 | 7.9538 | 0.9685 | 23.61 | 11.42 |
| consumption | augmented_open_meteo_missing | total | 87 | 6.0621 | 17.2637 | 0.9986 | 4.01 | 2.33 |
| consumption | augmented_open_meteo_missing | residencial | 87 | 1.6362 | 4.3441 | 0.9993 | 2.45 | 1.80 |
| consumption | augmented_open_meteo_missing | servicios | 87 | 4.4949 | 13.2058 | 0.9978 | 5.55 | 2.86 |
| consumption | augmented_open_meteo_missing | industria | 87 | 1.3978 | 5.5541 | 0.9694 | 21.74 | 12.16 |
| eolica | observed_current_weather | eolica | 40 | 1004.0911 | 1298.3863 | 0.7649 | 47.52 | 25.76 |
| eolica | augmented_open_meteo_missing | eolica | 88 | 929.0582 | 1216.2333 | 0.7937 | 47.75 | 23.84 |

## Lectura

- Si `augmented_open_meteo_missing` empeora poco o mejora, compensa por cobertura territorial.
- Si empeora bastante, mantener Open-Meteo como fuente separada y calibrar por isla/municipio antes de entrenar definitivo.
- La comparacion no es perfecta porque el escenario ampliado predice municipios adicionales, no exactamente el mismo panel.

## Reentrenamiento oficial posterior

Metricas test tras promover los datasets ampliados a `tfc-model/data` y reentrenar los scripts oficiales:

| Modelo oficial | Target | MAE | RMSE | R2 | MAPE | WMAPE |
|---|---|---:|---:|---:|---:|---:|
| consumo XGBoost | total | 6.4818 | 18.2288 | 0.9984 | 4.28 | 2.50 |
| consumo XGBoost | residencial | 1.5381 | 4.0956 | 0.9994 | 2.87 | 1.69 |
| consumo XGBoost | servicios | 5.0164 | 14.0631 | 0.9975 | 5.99 | 3.19 |
| consumo XGBoost | industria | 1.3147 | 5.4946 | 0.9700 | 22.25 | 11.45 |
| eolica HGB | eolica | 653.7800 | 853.1893 | 0.8985 | 29.82 | 16.77 |

Decision: se mantiene el dataset ampliado porque cubre 87/87 municipios ISTAC y la degradacion en consumo es moderada. Eolica mejora con la agregacion territorial ampliada.

## Estado final aplicado

- Los datasets oficiales de `tfc-model/data` ya son los ampliados.
- Los modelos oficiales ya fueron reentrenados sobre esos datasets.
- Las metricas, predicciones e imagenes de validacion oficiales fueron regeneradas por los scripts de entrenamiento.
- La app/API se adapto a la nueva cobertura: 87 municipios en metadatos, prediccion y coordenadas Open-Meteo.
- El smoke test de API carga todos los modelos sin fallos.

## Archivos generados

- `outputs/weather_daily_municipal_open_meteo_missing.csv`
- `outputs/weather_daily_municipal_augmented_full.csv`
- `outputs/final_demand_consumption_dataset_augmented_open_meteo.csv`
- `outputs/final_renewable_generation_dataset_augmented_open_meteo.csv`
- `tfc-model/src/evaluation/open_meteo_augmented/open_meteo_augmented_model_metrics.csv`

En `tfc-model/data` se conservan solo los nombres oficiales:

- `final_demand_consumption_dataset.csv`
- `final_renewable_generation_dataset.csv`
