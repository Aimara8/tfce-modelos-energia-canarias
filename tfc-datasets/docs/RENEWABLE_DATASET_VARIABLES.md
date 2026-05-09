# Variables del dataset de energias renovables

Archivo principal:
- `outputs/final_renewable_generation_dataset.csv`

## 1. Estructura general

El dataset esta organizado por **dia**. Cada fila representa un dia en Canarias.

Las variables se dividen en 3 bloques:
- fecha
- generacion renovable REE
- meteorologia agregada de Canarias

---

## 2. Variable temporal

### `date`
- Fecha del registro en formato `YYYY-MM-DD`.
- Es la clave principal para unir este dataset con otras series diarias.

---

## 3. Variables de generacion renovable de REE

Estas variables vienen de REE y aparecen en pares:
- `*_value`: energia absoluta de la tecnologia en ese dia
- `*_pct`: peso de esa tecnologia dentro del total renovable diario

### Total renovable

### `ree_generacion_renovable_value`
- Generacion renovable total del dia en Canarias.
- Es la suma del bloque renovable publicado por REE.
- Sirve como referencia para interpretar todos los porcentajes `*_pct`.

## Tecnologias renovables

### `ree_eolica_value`
- Energia eolica del dia.

### `ree_eolica_pct`
- Proporcion de la eolica respecto al total renovable del dia.

### `ree_solar_fotovoltaica_value`
- Energia solar fotovoltaica del dia.

### `ree_solar_fotovoltaica_pct`
- Proporcion de la solar fotovoltaica respecto al total renovable del dia.

### `ree_hidraulica_value`
- Energia hidraulica del dia.

### `ree_hidraulica_pct`
- Proporcion de la hidraulica respecto al total renovable del dia.

### `ree_hidroeolica_value`
- Energia hidroeolica del dia.
- En Canarias esta variable puede ser especialmente relevante por la combinacion de almacenamiento hidraulico y viento.

### `ree_hidroeolica_pct`
- Proporcion de la hidroeolica respecto al total renovable del dia.

### `ree_otras_renovables_value`
- Energia agrupada por REE como "otras renovables".
- Puede incluir tecnologias renovables con menor peso relativo o agregadas por la fuente.

### `ree_otras_renovables_pct`
- Proporcion de "otras renovables" respecto al total renovable del dia.

---

## 4. Variables meteorologicas agregadas

Estas variables no son municipales. Son una **agregacion diaria para Canarias** construida a partir de la meteorologia limpia disponible.

### Cobertura meteorologica

### `canarias_weather_municipality_count`
- Numero de municipios que aportaron algun dato meteorologico ese dia.
- Sirve para medir cobertura territorial.
- Puede ayudarte a filtrar dias con menor representatividad geografica.

### `canarias_weather_station_count`
- Numero total de estaciones consideradas en la agregacion de ese dia.
- Sirve para medir apoyo observacional de la media agregada.

### Temperatura

### `temp_avg_c`
- Temperatura media diaria agregada de Canarias.

### `temp_max_c`
- Temperatura maxima diaria agregada.

### `temp_min_c`
- Temperatura minima diaria agregada.

### Humedad y punto de rocio

### `humidity_avg_pct`
- Humedad relativa media diaria.

### `dew_point_avg_c`
- Punto de rocio medio diario.

### Presion

### `pressure_avg_hpa`
- Presion atmosferica media diaria.

### Precipitacion

### `precip_intensity_avg_mm`
- Intensidad media de precipitacion.

### `rain_daily_mm`
- Precipitacion diaria acumulada agregada.

### Viento

### `wind_speed_avg_ms`
- Velocidad media diaria del viento.

### `wind_speed_max_ms`
- Velocidad maxima diaria del viento.

### `wind_speed_sdev_ms`
- Variabilidad de la velocidad del viento.

### `wind_dir_avg_deg`
- Direccion media del viento en grados.

### `wind_dir_max_deg`
- Direccion asociada al maximo registrado segun la fuente agregada.

### `wind_dir_sdev_deg`
- Variabilidad de la direccion del viento.

### Trazabilidad

### `weather_data_source`
- Texto de apoyo sobre el origen de la meteorologia del dataset final.

---

## 5. Relaciones utiles entre variables

### Relacion base de porcentajes

Para cada dia:

`ree_tecnologia_pct = ree_tecnologia_value / ree_generacion_renovable_value`

Ejemplos:
- `ree_eolica_pct = ree_eolica_value / ree_generacion_renovable_value`
- `ree_solar_fotovoltaica_pct = ree_solar_fotovoltaica_value / ree_generacion_renovable_value`

### Relacion entre tecnologias

Si quieres estudiar la composicion del mix renovable diario, compara:
- eolica frente a solar fotovoltaica
- hidraulica frente a hidroeolica
- otras renovables como bloque residual

### Relacion con meteorologia

Relaciones esperables:
- mas `wind_speed_avg_ms` puede favorecer `ree_eolica_value`
- mas radiacion no esta directamente en el dataset, pero temperaturas despejadas y ciertas condiciones pueden acompañar mayor `ree_solar_fotovoltaica_value`
- lluvia, humedad y viento pueden ayudar a interpretar dias atipicos

---

## 6. Recomendacion practica

Si quieres un dataset mas simple para modelar, normalmente bastaria con:
- `date`
- `ree_generacion_renovable_value`
- valores por tecnologia `*_value`
- algunas meteorologicas clave como `temp_avg_c`, `wind_speed_avg_ms`, `wind_speed_max_ms`, `humidity_avg_pct`, `rain_daily_mm`

Los campos de conteo:
- `canarias_weather_municipality_count`
- `canarias_weather_station_count`

no son obligatorios, pero pueden ser utiles como indicadores de calidad del dato.
