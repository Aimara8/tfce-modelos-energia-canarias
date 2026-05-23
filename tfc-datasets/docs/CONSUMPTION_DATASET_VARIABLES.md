# Variables del dataset de consumo electrico

Archivo principal:
- `outputs/final_demand_consumption_dataset_augmented_open_meteo.csv`
- copia oficial para modelos: `tfc-model/data/final_demand_consumption_dataset.csv`

## 1. Estructura general

El dataset esta organizado por **municipio y dia**. Cada fila representa un municipio ISTAC de Canarias en una fecha concreta.

Cobertura actual: **87/87 municipios ISTAC**. La meteorologia observada mantiene prioridad y Open-Meteo rellena los municipios sin cobertura de estacion.

Las variables se dividen en 4 bloques:
- identificacion territorial
- fecha
- consumo electrico
- meteorologia municipal asociada

---

## 2. Variables identificadoras

### `municipality`
- Nombre del municipio.
- Es la clave territorial principal del dataset.

### `date`
- Fecha del registro en formato `YYYY-MM-DD`.
- Junto con `municipality` forma la clave principal de cada fila.

---

## 3. Variables de consumo electrico

Estas variables proceden de ISTAC y estan expresadas en **MWh por dia**.

### `demand_total_mwh`
- Consumo electrico total diario del municipio.
- Es la variable mas natural si quieres modelar demanda agregada.

### `demand_industria_mwh`
- Consumo electrico diario del sector industrial.
- Puede ayudar a captar municipios con mayor peso productivo.

### `demand_residencial_mwh`
- Consumo electrico diario del sector residencial.
- Suele estar mas relacionado con habitos de hogares, temperatura y estacionalidad.

### `demand_servicios_mwh`
- Consumo electrico diario del sector servicios.
- Puede reflejar actividad comercial, turistica y terciaria.

---

## 4. Variables meteorologicas municipales

Estas variables vienen del procesamiento meteorologico diario y se unen a nivel de municipio y fecha.

### Cobertura meteorologica

### `weather_station_count`
- Numero de estaciones meteorologicas que aportaron datos utiles para ese municipio y ese dia.
- Sirve como indicador de cobertura y apoyo observacional.

### `weather_data_source`
- Origen de la meteorologia de la fila.
- Valores esperados: `observed_station` u `open_meteo_historical_missing_municipality`.
- Permite auditar si una prediccion historica viene de estacion real o de relleno Open-Meteo.

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
- Intensidad media de precipitacion.

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
- Direccion media del viento en grados.

### `wind_dir_max_deg`
- Direccion maxima registrada segun la agregacion disponible.

### `wind_dir_sdev_deg`
- Variabilidad de la direccion del viento.

---

## 5. Relaciones utiles entre variables

### Relacion entre total y sectores

En terminos conceptuales:

`demand_total_mwh ~= demand_industria_mwh + demand_residencial_mwh + demand_servicios_mwh`

Puede no coincidir de forma perfecta si la fuente original incluye redondeos, clasificaciones parciales o algun ajuste estadistico.

### Relacion con meteorologia

Relaciones esperables:
- temperaturas mas altas o mas bajas pueden modificar `demand_residencial_mwh`
- episodios meteorologicos adversos pueden alterar consumo residencial y de servicios
- municipios turisticos pueden mostrar patrones distintos en `demand_servicios_mwh`

### Calidad del dato meteorologico

Si `weather_station_count` es bajo:
- la meteorologia disponible puede ser menos representativa
- puede convenirte usar esa variable como predictor o como filtro de calidad

---

## 6. Recomendacion practica

Si quieres un dataset simple para modelar demanda, normalmente bastaria con:
- `municipality`
- `date`
- `demand_total_mwh`
- `temp_avg_c`
- `humidity_avg_pct`
- `rain_daily_mm`
- `wind_speed_avg_ms`
- `weather_station_count`

Si quieres un analisis mas explicativo, conserva tambien:
- `demand_industria_mwh`
- `demand_residencial_mwh`
- `demand_servicios_mwh`

porque ayudan a entender que parte del consumo responde a distintos perfiles de actividad.
