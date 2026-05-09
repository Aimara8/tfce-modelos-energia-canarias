# tfc-model

Código para el entrenamiento y evaluación de modelos de predicción energética usados en este proyecto.

## Estructura

- `src/training/`: scripts y puntos de entrada para entrenar modelos.
- `src/models/`: modelos entrenados y artefactos resultantes del entrenamiento.
- `src/evaluation/`: métricas, validación y análisis de resultados.
- `data/`: conjuntos de datos locales para entrenamiento. Los archivos CSV grandes deben estar ignorados en Git.

## Datos

Esta carpeta espera los datasets procesados generados en `tfc-datasets/outputs/`.

Archivos recomendados localmente:

- `data/final_demand_consumption_dataset.csv`
- `data/final_renewable_generation_dataset.csv`

## Notas

- Mantener los datasets pesados fuera de Git.
- Documentar experimentos y parámetros junto al código de entrenamiento.
