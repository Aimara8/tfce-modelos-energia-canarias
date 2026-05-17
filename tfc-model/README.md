# `tfc-model`

Entrenamiento y evaluacion de modelos predictivos para el TFC de energia en Canarias.

## Vista Rapida

- `Consumo electrico`: modelado sectorial con `XGBoost`
- `Generacion renovable`: modelado especifico por tecnologia
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
- `src/training/Renewable_Energy_Generation/XGBoost.py`
  Script comparativo ampliado entre modelos lineales y de arboles.

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

### Generacion renovable

- `Eolica`: el mejor rendimiento actual lo da `HistGradientBoosting`
- `Solar`: con las variables disponibles, los modelos lineales son mas estables
- la evaluacion se separa del modelo para mantener una estructura mas limpia

### Consumo electrico

- se mantiene el enfoque sectorial con un modelo por tipo de demanda
- las salidas ya no se guardan fuera de `src`

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
