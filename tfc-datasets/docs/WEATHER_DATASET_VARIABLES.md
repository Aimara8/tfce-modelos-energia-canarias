# Variables del dataset meteorologico municipal

Archivo principal:
- `outputs/weather_daily_municipal_clean.csv`

## 1. Estructura general

El dataset esta organizado por **municipio y dia**. Cada fila representa la meteorologia diaria agregada de un municipio.

Las variables se dividen en 3 bloques:
- identificacion territorial
- fecha
- meteorologia diaria agregada

---

## 2. Variables identificadoras

### `municipality`
- Nombre del municipio.
- Es la clave territorial principal del dataset meteorologico.

### `date`
- Fecha del registro en formato `YYYY-MM-DD`.
- Junto con `municipality` forma la clave principal de cada fila.

---

## 3. Variables meteorologicas

Estas variables proceden del procesamiento de observaciones meteorologicas y se agregan por municipio y dia.

### Cobertura

### `weather_station_count`
- Numero de estaciones que aportaron datos utiles al municipio en ese dia.
- Es importante para medir solidez y cobertura del dato agregado.

### Temperatura

### `temp_avg_c`
- Temperatura media diaria del municipio.

### `temp_max_c`
- Temperatura maxima diaria del municipio.

### `temp_min_c`
- Temperatura minima diaria del municipio.

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
- Intensidad media de precipitacion para el dia.

### `rain_daily_mm`
- Precipitacion diaria acumulada.

### Viento

### `wind_speed_avg_ms`
- Velocidad media diaria del viento.

### `wind_speed_max_ms`
- Velocidad maxima diaria del viento.

### `wind_speed_sdev_ms`
- Variabilidad de la velocidad del viento.

### `wind_dir_avg_deg`
- Direccion media diaria del viento en grados.

### `wind_dir_max_deg`
- Direccion maxima registrada segun la agregacion disponible.

### `wind_dir_sdev_deg`
- Variabilidad de la direccion del viento.

---

## 4. Como interpretar valores vacios

Si una variable aparece vacia:
- no significa necesariamente valor cero
- normalmente significa que no hubo observacion valida suficiente para ese municipio y ese dia en esa variable

Esto es importante sobre todo en:
- `rain_daily_mm`
- direccion del viento
- variables con menor cobertura observacional

---

## 5. Relaciones utiles entre variables

### Consistencia termica

Normalmente deberia cumplirse:

`temp_min_c <= temp_avg_c <= temp_max_c`

Si detectas excepciones, conviene revisarlas como posibles incidencias de agregacion o calidad de origen.

### Relacion humedad y punto de rocio

Suele existir una relacion estrecha entre:
- `humidity_avg_pct`
- `dew_point_avg_c`

Juntas ayudan a describir mejor la sensacion ambiental que la temperatura por si sola.

### Relacion de viento

Las variables:
- `wind_speed_avg_ms`
- `wind_speed_max_ms`
- `wind_speed_sdev_ms`

permiten distinguir dias estables de dias con rachas o variabilidad elevada.

### Cobertura y fiabilidad

`weather_station_count` no es una variable meteorologica directa, pero si una señal clave de calidad:
- mas estaciones suele implicar una agregacion mas robusta
- menos estaciones puede implicar mayor sensibilidad a ruido local

---

## 6. Relacion con otros datasets del proyecto

Este dataset se usa como base para dos cosas:
- alimentar el dataset final de consumo electrico a nivel municipal
- construir una agregacion diaria de Canarias para el dataset final de renovables

Por eso actua como pieza intermedia central del pipeline meteorologico.

---

## 7. Recomendacion practica

Si quieres una version compacta para analisis exploratorio, normalmente bastaria con:
- `municipality`
- `date`
- `temp_avg_c`
- `humidity_avg_pct`
- `rain_daily_mm`
- `wind_speed_avg_ms`
- `weather_station_count`

Si vas a modelar o estudiar episodios extremos, conviene conservar tambien:
- `temp_max_c`
- `temp_min_c`
- `wind_speed_max_ms`
- `wind_speed_sdev_ms`
- `pressure_avg_hpa`
