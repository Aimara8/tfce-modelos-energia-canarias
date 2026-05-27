# TFCE - Modelos de Energía en Canarias

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-FF6600)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/Model-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

Repositorio del TFCE centrado en el análisis, modelado e inferencia de demanda eléctrica municipal y generación renovable en Canarias.

El proyecto combina datos oficiales de consumo eléctrico, generación renovable y meteorología para construir datasets reproducibles, entrenar modelos predictivos y exponer los resultados mediante API y aplicación interactiva.

---

## Demo interactiva

[![Abrir en Streamlit](https://img.shields.io/badge/Abrir%20demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://tfce-modelos-energia-canarias.streamlit.app/)

Enlace directo: [tfce-modelos-energia-canarias.streamlit.app](https://tfce-modelos-energia-canarias.streamlit.app/)

---

## Objetivo

El objetivo principal es desarrollar una solución completa de predicción energética para Canarias, cubriendo dos problemas:

| Área | Predicción | Enfoque |
|---|---|---|
| Consumo eléctrico | Demanda municipal diaria por sector | Modelos XGBoost por sector |
| Generación renovable | Generación eólica diaria | HistGradientBoosting frente a Ridge |

El proyecto prioriza tres aspectos:

- Uso de datos oficiales y trazables.
- Cobertura territorial completa para los 87 municipios ISTAC de Canarias.
- Separación clara entre datos, entrenamiento, evaluación, API e interfaz.

---

## Datos utilizados

El pipeline integra tres bloques principales de información:

| Fuente | Uso en el proyecto |
|---|---|
| ISTAC | Demanda eléctrica municipal por sectores |
| REE | Generación renovable diaria por tecnología |
| Meteorología observada + Open-Meteo | Variables meteorológicas para entrenamiento e inferencia |

La ampliación con Open-Meteo se usa únicamente para municipios sin cobertura meteorológica observada. Las estaciones reales mantienen prioridad cuando existen datos disponibles.

### Cobertura actual

| Dataset | Periodo | Filas | Cobertura |
|---|---:|---:|---:|
| Consumo eléctrico | 2020-01-01 a 2025-06-30 | 165.106 | 87/87 municipios ISTAC |
| Renovables | 2020-01-01 a 2025-06-30 | 2.008 | Serie diaria agregada Canarias |

---

## Resultados oficiales

Métricas de test tras reentrenar los modelos oficiales con los datasets ampliados.

| Modelo | Target | MAE | RMSE | R² | MAPE | WMAPE |
|---|---|---:|---:|---:|---:|---:|
| XGBoost | Consumo total | 6.48 | 18.23 | 0.9984 | 4.28% | 2.50% |
| XGBoost | Residencial | 1.54 | 4.10 | 0.9994 | 2.87% | 1.69% |
| XGBoost | Servicios | 5.02 | 14.06 | 0.9975 | 5.99% | 3.19% |
| XGBoost | Industria | 1.31 | 5.49 | 0.9700 | 22.25% | 11.45% |
| HistGradientBoosting | Eólica | 653.78 | 853.19 | 0.8985 | 29.82% | 16.77% |

Las métricas y gráficas se encuentran en:

```text
tfc-model/src/evaluation/
```
## Estructura del repositorio

```text
tfce-modelos-energia-canarias/
├── tfc-datasets/
│   ├── inputs/                  # Fuentes originales
│   ├── scripts/                 # Scripts de construcción y ampliación de datasets
│   ├── outputs/                 # Datasets intermedios y finales generados
│   └── docs/                    # Documentación del pipeline y variables
│
├── tfc-model/
│   ├── data/                    # Datasets finales usados por entrenamiento e inferencia
│   ├── src/
│   │   ├── training/            # Scripts de entrenamiento
│   │   ├── models/              # Modelos serializados
│   │   ├── evaluation/          # Métricas, predicciones y visualizaciones
│   │   ├── api/                 # API FastAPI
│   │   └── client/              # Cliente Streamlit
│   ├── app/                     # API local alternativa con frontend estático
│   ├── requirements.txt
│   └── TRAINING_RUNS.md
│
├── requirements.txt
└── README.md
```
## Componentes principales

### `tfc-datasets`

Contiene el pipeline de preparación de datos.

Incluye:

- Integración de demanda eléctrica municipal.
- Procesamiento de generación renovable diaria.
- Unión con variables meteorológicas.
- Ampliación territorial con Open-Meteo.
- Documentación de variables y decisiones del pipeline.

Script principal de ampliación:

```bash
python tfc-datasets/scripts/build_open_meteo_augmented_dataset.py
```
### `tfc-model`

Contiene la parte de modelado, evaluación, API e interfaz interactiva del proyecto.

Incluye:

- Entrenamiento de modelos predictivos.
- Evaluación con métricas y gráficas.
- Modelos serializados para inferencia.
- API con FastAPI.
- Cliente interactivo con Streamlit.

Modelos principales:

| Problema | Script | Salida |
|---|---|---|
| Consumo eléctrico | `src/training/Energy_Consumption/XGBoost.py` | 4 modelos XGBoost |
| Generación eólica | `src/training/Renewable_Energy_Generation/HistGradientBoosting_VS_Ridge.py` | Modelo HistGradientBoosting y comparativa Ridge |

Los modelos entrenados se guardan en:

```text
tfc-model/src/models/
```

Las métricas, predicciones y gráficas se guardan en:

```text
tfc-model/src/evaluation/
```

---

### API FastAPI

La API permite consultar predicciones de consumo eléctrico municipal y generación eólica diaria.

Arranque local:

```bash
cd tfc-model
python -m uvicorn src.api.main:app --reload --port 8000
```

Endpoints principales:

```text
GET  /health
GET  /metadata
GET  /dashboard/consumo
GET  /dashboard/eolica
POST /predict/consumo
POST /predict/eolica
```

---

### Cliente Streamlit

Interfaz interactiva para explorar predicciones, métricas y datos del proyecto.

Arranque local:

```bash
cd tfc-model
python -m streamlit run src/client/app.py
```

Por defecto, la aplicación estará disponible en:

```text
http://localhost:8501
```

---

## Instalación local

### Requisitos

- Python 3.8 o superior
- pip
- Entorno virtual recomendado

### Configuración

```bash
git clone https://github.com/Aimara8/tfce-modelos-energia-canarias.git
cd tfce-modelos-energia-canarias

python -m venv .venv
```

Activar entorno en Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Activar entorno en Linux/macOS:

```bash
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## Entrenamiento de modelos

Desde la carpeta `tfc-model`:

```bash
cd tfc-model
python src/training/Energy_Consumption/XGBoost.py
python src/training/Renewable_Energy_Generation/HistGradientBoosting_VS_Ridge.py
```

Las salidas se generan en:

```text
src/models/
src/evaluation/
```

---

## Flujo de trabajo

```text
Fuentes originales
ISTAC + REE + meteorología
        ↓
tfc-datasets
limpieza, unión y ampliación Open-Meteo
        ↓
datasets finales
87/87 municipios ISTAC
        ↓
tfc-model
entrenamiento y evaluación
        ↓
modelos serializados
métricas, predicciones y gráficas
        ↓
API FastAPI + cliente Streamlit
inferencia y visualización
```

---

## Documentación

Documentación de datos:

```text
tfc-datasets/docs/Context.md
tfc-datasets/docs/CONSUMPTION_DATASET_VARIABLES.md
tfc-datasets/docs/RENEWABLE_DATASET_VARIABLES.md
tfc-datasets/docs/WEATHER_DATASET_VARIABLES.md
tfc-datasets/docs/OPEN_METEO_AUGMENTATION.md
```

Documentación de modelos:

```text
tfc-model/README.md
tfc-model/TRAINING_RUNS.md
tfc-model/src/api/README.md
```

---

## Tecnologías

| Categoría | Herramientas |
|---|---|
| Lenguaje principal | Python |
| Datos | pandas, NumPy |
| Machine Learning | XGBoost, scikit-learn |
| Visualización | matplotlib, seaborn, Plotly |
| API | FastAPI, Uvicorn, Pydantic |
| Aplicación | Streamlit |
| Utilidades | joblib, requests, psutil |

---

## Estado del proyecto

- Datasets oficiales ampliados con Open-Meteo.
- Cobertura de consumo para 87/87 municipios ISTAC.
- Modelos oficiales reentrenados.
- Métricas, predicciones y visualizaciones regeneradas.
- API y cliente adaptados a la cobertura territorial completa.

---

## Autoría

Proyecto desarrollado por **Aimara** como parte del TFCE.

---

**Última actualización:** 2026-05-26
