# `data`

Carpeta de entrada para los datasets finales usados por los scripts de entrenamiento.

## Que Debe Haber Aqui

- `final_demand_consumption_dataset.csv`
- `final_renewable_generation_dataset.csv`

## Funcion De Estos Archivos

- alimentar los scripts de `src/training/`
- servir como version estable del dataset ya procesado
- separar claramente datos de entrada de modelos y evaluaciones

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
