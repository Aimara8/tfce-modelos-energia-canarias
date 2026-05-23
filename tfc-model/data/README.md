# `data`

Carpeta de entrada para los datasets finales usados por los scripts de entrenamiento.

## Que Debe Haber Aqui

- `final_demand_consumption_dataset.csv`
- `final_renewable_generation_dataset.csv`

Estos son los datasets oficiales actuales para entrenamiento e inferencia. Ya incorporan la ampliacion Open-Meteo:

- consumo: 87/87 municipios ISTAC
- renovables: meteorologia agregada diaria de Canarias con cobertura ampliada
- fuente meteorologica: estaciones reales con prioridad; Open-Meteo solo para municipios/dias sin dato observado

## Funcion De Estos Archivos

- alimentar los scripts de `src/training/`
- servir como version estable del dataset ya procesado
- separar claramente datos de entrada de modelos y evaluaciones
- evitar copias rollback o variantes temporales dentro de `data/`

## Flujo Recomendado

```text
dataset procesado -> data/ -> src/training/ -> src/models/ + src/evaluation/
```

## Buenas Practicas

- no mezclar aqui modelos ni graficas
- no guardar copias intermedias innecesarias
- mantener nombres consistentes y descriptivos
- evitar subir archivos muy pesados si no son necesarios

## Relacion Con El Proyecto

- `data/` contiene entradas
- `src/models/` contiene artefactos entrenados
- `src/evaluation/` contiene metricas, predicciones y visualizaciones
