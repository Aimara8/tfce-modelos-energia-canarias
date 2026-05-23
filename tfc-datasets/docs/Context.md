# Contexto del Proyecto

Este proyecto construye **dos datasets finales** para modelos predictivos en Canarias:

1. **Demanda de consumo electrico**
2. **Generacion de energias renovables**

La idea no es tener un unico dataset gigante, sino dos tablas finales limpias, coherentes y listas para modelado.

---

## 1. Objetivo de cada dataset

### Dataset de demanda

Archivo final:
- `outputs/final_demand_consumption_dataset_augmented_open_meteo.csv`
- copia oficial de entrenamiento: `tfc-model/data/final_demand_consumption_dataset.csv`

Objetivo:
- modelar la demanda electrica diaria
- trabajar a nivel **municipal**
- conservar meteorologia real cuando existe
- rellenar municipios sin cobertura observada con Open-Meteo historico

Cada fila representa:
- un **municipio**
- un **dia**

### Dataset de renovables

Archivo final:
- `outputs/final_renewable_generation_dataset_augmented_open_meteo.csv`
- copia oficial de entrenamiento: `tfc-model/data/final_renewable_generation_dataset.csv`

Objetivo:
- modelar la generacion renovable diaria
- trabajar a nivel **Canarias como sistema electrico**
- combinar generacion renovable de REE con meteorologia agregada diaria

Cada fila representa:
- **Canarias**
- un **dia**
- cada tecnologia renovable aparece como columna

---

## 2. Fuentes de datos

### Consumo electrico

- **ISTAC**
- fichero base: `inputs/istac/dataset-ISTAC_C00022A_000005_1.11_20260426202346.csv`

Se usa para:
- demanda total
- demanda por sectores

### Meteorologia

- **Sistema de Observacion Meteorologica de Canarias**
- ficheros base:
  - `inputs/weather/estaciones.csv`
  - `inputs/weather/observaciones_2020.csv`
  - `inputs/weather/observaciones_2021.csv`
  - `inputs/weather/observaciones_2022.csv`
  - `inputs/weather/observaciones_2023.csv`
  - `inputs/weather/observaciones_2024.csv`
  - `inputs/weather/observaciones_2025.csv`

Se usa para:
- temperatura
- humedad
- presion
- precipitacion
- viento

### Open-Meteo

- **Open-Meteo Historical Weather API**
- se usa solo para municipios sin cobertura meteorologica observada
- las estaciones reales mantienen prioridad
- las coordenadas municipales quedan fijadas en `outputs/municipality_coordinates_open_meteo.csv`

### Generacion renovable

- **REE / REData**
- fuente usada: backend oficial de datos de REE

Se usa para:
- generacion renovable total diaria
- detalle diario por tecnologia renovable
- porcentaje de cada tecnologia dentro del total renovable del dia

---

## 3. Scripts que quedan en uso

### `scripts/build_demand_weather_dataset.js`

Funcion:
- limpia demanda ISTAC
- limpia meteorologia
- mapea estaciones a municipios
- agrega meteorologia a nivel municipal diario
- genera el dataset combinado intermedio de demanda + clima
- admite tanto la nueva estructura `inputs/...` como las rutas antiguas en raiz

Salidas:
- `outputs/stations_clean.csv`
- `outputs/istac_daily_municipal_clean.csv`
- `outputs/weather_daily_municipal_clean.csv`
- `outputs/demand_weather_daily_municipal.csv`

### `scripts/fetch_ree_renewables.js`

Funcion:
- descarga la estructura de generacion renovable diaria de Canarias desde REE

Salidas:
- `outputs/ree_renewables_canarias_daily_wide.csv`

### `scripts/build_final_datasets.js`

Funcion:
- construye el dataset final de demanda a partir del dataset intermedio
- elimina las filas donde **todos** los campos climaticos estan vacios
- construye el dataset final de renovables uniendo REE con meteorologia agregada diaria de Canarias

Salidas:
- `outputs/final_demand_consumption_dataset.csv`
- `outputs/final_renewable_generation_dataset.csv`

### `scripts/build_open_meteo_augmented_dataset.py`

Funcion:
- detecta municipios ISTAC sin meteorologia observada
- usa coordenadas municipales estables para consultar o reconstruir Open-Meteo
- genera meteorologia diaria para municipios faltantes
- combina estaciones reales + Open-Meteo con prioridad para estaciones reales
- construye los datasets finales ampliados de consumo y renovables
- copia los datasets ampliados como oficiales en `tfc-model/data`

Salidas:
- `outputs/weather_daily_municipal_open_meteo_missing.csv`
- `outputs/weather_daily_municipal_augmented_full.csv`
- `outputs/final_demand_consumption_dataset_augmented_open_meteo.csv`
- `outputs/final_renewable_generation_dataset_augmented_open_meteo.csv`

---

## 4. Datasets intermedios y finales

### Intermedios

#### `outputs/istac_daily_municipal_clean.csv`
- demanda diaria municipal limpia

#### `outputs/weather_daily_municipal_clean.csv`
- meteorologia diaria municipal limpia
- corresponde a estaciones reales

#### `outputs/demand_weather_daily_municipal.csv`
- union intermedia de demanda y meteorologia
- aun contiene filas sin cobertura meteorologica

#### `outputs/weather_daily_municipal_augmented_full.csv`
- meteorologia diaria municipal ampliada
- combina estaciones reales y Open-Meteo para municipios faltantes

#### `outputs/ree_renewables_canarias_daily_wide.csv`
- generacion renovable diaria en formato ancho
- una fila por fecha
- una columna por tecnologia y porcentaje asociado

### Finales

#### `outputs/final_demand_consumption_dataset_augmented_open_meteo.csv`
- dataset final ampliado para el modelo de demanda
- cubre 87/87 municipios ISTAC
- incluye `weather_data_source` para distinguir estacion real frente a Open-Meteo

#### `outputs/final_renewable_generation_dataset_augmented_open_meteo.csv`
- dataset final ampliado para el modelo eolico
- incluye generacion REE y meteorologia agregada diaria de Canarias

---

## 5. Medidas y unidades

### Demanda electrica

Variables principales:
- `demand_total_mwh`
- `demand_industria_mwh`
- `demand_residencial_mwh`
- `demand_servicios_mwh`

Unidad:
- **MWh** por dia

### Temperatura

Variables:
- `temp_avg_c`
- `temp_max_c`
- `temp_min_c`
- `dew_point_avg_c`

Unidad:
- **grados Celsius (C)**

### Presion

Variable:
- `pressure_avg_hpa`

Unidad:
- **hPa**

### Precipitacion

Variables:
- `precip_intensity_avg_mm`
- `rain_daily_mm`

Unidad:
- **mm**

### Humedad

Variable:
- `humidity_avg_pct`

Unidad:
- **porcentaje (%)**

### Viento

Variables:
- `wind_speed_avg_ms`
- `wind_speed_max_ms`
- `wind_speed_sdev_ms`

Unidad:
- **m/s**

Variables de direccion:
- `wind_dir_avg_deg`
- `wind_dir_max_deg`
- `wind_dir_sdev_deg`

Unidad:
- **grados (deg)**

### Generacion renovable REE

Variables tipo energia:
- `ree_*_value`

Unidad:
- la energia diaria publicada por REE para cada tecnologia y para el total renovable

Variables tipo proporcion:
- `ree_*_pct`

Unidad:
- proporcion del total renovable diario, entre **0 y 1**

---

## 6. Logica actual del filtrado

### Demanda

El dataset final de demanda oficial:
- parte del dataset combinado intermedio
- mantiene estaciones reales como fuente prioritaria
- rellena los municipios ISTAC sin meteorologia observada con Open-Meteo historico

Resultado actual:
- **165.106 registros**
- **87 municipios**
- periodo: `2020-01-01` a `2025-06-30`

### Renovables

El dataset final de renovables:
- usa REE como fuente principal
- agrega la meteorologia municipal a una sola serie diaria de Canarias
- mantiene variables de cobertura:
  - `canarias_weather_municipality_count`
  - `canarias_weather_station_count`

Resultado actual:
- **2.008 registros**
- cobertura meteorologica diaria ampliada: 69 a 87 municipios segun disponibilidad por fecha
- periodo: `2020-01-01` a `2025-06-30`

---

## 7. Documentacion adicional

- `docs/RENEWABLE_DATASET_VARIABLES.md`
- `docs/CONSUMPTION_DATASET_VARIABLES.md`
- `docs/WEATHER_DATASET_VARIABLES.md`
- `docs/OPEN_METEO_AUGMENTATION.md`

Estos archivos explican con mas detalle las variables de renovables, consumo, clima, ampliacion Open-Meteo y la relacion entre ellas.
