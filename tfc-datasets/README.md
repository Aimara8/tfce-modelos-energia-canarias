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

## Entradas principales

- `inputs/istac/`: demanda electrica ISTAC
- `inputs/weather/`: estaciones y observaciones meteorologicas

## Documentacion

- `docs/Context.md`
- `docs/RENEWABLE_DATASET_VARIABLES.md`
- `docs/CONSUMPTION_DATASET_VARIABLES.md`
- `docs/WEATHER_DATASET_VARIABLES.md`
