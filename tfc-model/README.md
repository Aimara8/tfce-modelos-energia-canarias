# `tfc-model`

Entrenamiento y evaluacion de modelos predictivos para el TFC de energia en Canarias.

## Vista Rapida

- `Consumo electrico`: modelado sectorial con `XGBoost`
- `Generacion renovable`: modelo eolico diario con `HistGradientBoosting`
- `Datos actuales`: dataset ampliado con Open-Meteo para cubrir 87/87 municipios ISTAC
- `Modelos serializados`: dentro de `src/models/`
- `Metricas, predicciones y graficas`: dentro de `src/evaluation/`

## Estructura

```text
tfc-model/
├── data/
├── src/
│   ├── training/
│   │   ├── Energy_Consumption/
│   │   └── Renewable_Energy_Generation/
│   ├── models/
│   │   ├── consumption_energy_demand/
│   │   └── renewable_energy_generation/
│   └── evaluation/
│       ├── consumption_energy_demand/
│       └── renewable_energy_generation/
└── requirements.txt
```

## Scripts Principales

### Consumo electrico

- `src/training/Energy_Consumption/XGBoost.py`
  Entrena modelos por sector: `total`, `residencial`, `servicios` e `industria`.

### Generacion renovable

- `src/training/Renewable_Energy_Generation/HistGradientBoosting_VS_Ridge.py`
  Comparativa final para `eolica`.

## Convencion De Salidas

### Modelos

Se guardan en `src/models/...`

- `.pkl` para modelos de `scikit-learn`
- `.json` para modelos `XGBoost`
- imputadores y objetos auxiliares necesarios para inferencia

### Evaluacion

Se guarda en `src/evaluation/...`

- metricas `.csv`
- predicciones de test `.csv`
- graficas `.png`

## Estado Actual Del Proyecto

### Datos oficiales

- `data/final_demand_consumption_dataset.csv`: dataset oficial de consumo, ya ampliado con Open-Meteo.
- `data/final_renewable_generation_dataset.csv`: dataset oficial de renovables, con meteorologia agregada ampliada.
- Cobertura consumo: 87/87 municipios ISTAC.
- Open-Meteo solo rellena municipios sin estacion/meteorologia observada; las estaciones reales mantienen prioridad.

### Metricas oficiales actuales

Metricas de test tras reentrenar los scripts oficiales con el dataset ampliado:

| Modelo | Target | MAE | RMSE | R2 | MAPE | WMAPE |
|---|---|---:|---:|---:|---:|---:|
| consumo XGBoost | total | 6.4818 | 18.2288 | 0.9984 | 4.28 | 2.50 |
| consumo XGBoost | residencial | 1.5381 | 4.0956 | 0.9994 | 2.87 | 1.69 |
| consumo XGBoost | servicios | 5.0164 | 14.0631 | 0.9975 | 5.99 | 3.19 |
| consumo XGBoost | industria | 1.3147 | 5.4946 | 0.9700 | 22.25 | 11.45 |
| eolica HGB | eolica | 653.7800 | 853.1893 | 0.8985 | 29.82 | 16.77 |

### Generacion renovable

- `Eolica`: el modelo oficial actual es `HistGradientBoosting`.
- La evaluacion se guarda separada del modelo para mantener una estructura limpia.

### Consumo electrico

- se mantiene el enfoque sectorial con un modelo por tipo de demanda
- las salidas oficiales se guardan en `src/evaluation/consumption_energy_demand/`

## Instalacion

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Ejecucion

```powershell
python src/training/Energy_Consumption/XGBoost.py
python src/training/Renewable_Energy_Generation/HistGradientBoosting_VS_Ridge.py
```

Los tiempos y consumo de recursos medidos se documentan en `TRAINING_RUNS.md`.

## Idea Clave Del Repositorio

Cada script debe poder:

1. cargar datos procesados desde `data/`
2. entrenar con separacion temporal
3. guardar el modelo en `src/models/`
4. guardar la evaluacion en `src/evaluation/`

## Nota

Si anades nuevos entrenamientos, intenta mantener la misma convencion:

- modelos a `src/models/<problema>/`
- evaluacion a `src/evaluation/<problema>/`
