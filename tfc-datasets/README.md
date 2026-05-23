# tfc-datasets

Este directorio queda organizado en cuatro bloques:

- `inputs/`: fuentes originales descargadas o exportadas
- `scripts/`: scripts de construccion
- `outputs/`: datasets intermedios y finales generados
- `docs/`: documentacion del pipeline y variables

## Renovables REE

- `outputs/ree_renewables_canarias_daily_wide.csv`
  - una fila por fecha
  - una columna por tecnologia renovable y su porcentaje
  - es el unico formato intermedio que usa `scripts/build_final_datasets.js`
  - se mantiene solo el formato `wide` para reducir complejidad del pipeline y facilitar su explicacion

## Dataset ampliado Open-Meteo

El pipeline actual mantiene estaciones reales como fuente prioritaria y usa Open-Meteo solo para rellenar municipios sin cobertura meteorologica observada.

- Municipios con meteo observada inicial: 39
- Municipios anadidos con Open-Meteo: 48
- Cobertura final de consumo: 87/87 municipios ISTAC

Script oficial:

- `scripts/build_open_meteo_augmented_dataset.py`

Salidas principales:

- `outputs/weather_daily_municipal_open_meteo_missing.csv`
- `outputs/weather_daily_municipal_augmented_full.csv`
- `outputs/final_demand_consumption_dataset_augmented_open_meteo.csv`
- `outputs/final_renewable_generation_dataset_augmented_open_meteo.csv`

Informe:

- `docs/OPEN_METEO_AUGMENTATION.md`

## Entradas principales

- `inputs/istac/`: demanda electrica ISTAC
- `inputs/weather/`: estaciones y observaciones meteorologicas

## Documentacion

- `docs/Context.md`
- `docs/RENEWABLE_DATASET_VARIABLES.md`
- `docs/CONSUMPTION_DATASET_VARIABLES.md`
- `docs/WEATHER_DATASET_VARIABLES.md`
- `docs/OPEN_METEO_AUGMENTATION.md`
