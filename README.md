# TFCE - Modelos de Energía en Canarias

Repositorio TFCE centrado en el análisis de la demanda eléctrica y la generación renovable en Canarias, a partir de datos oficiales y meteorológicos.

## 🚀 Demo Interactiva

Prueba la aplicación en vivo:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tfce-modelos-energia-canarias.streamlit.app/)

## 📋 Descripción

Este proyecto desarrolla modelos predictivos para:

- **Consumo Eléctrico**: Modelado sectorial (total, residencial, servicios, industria) usando XGBoost
- **Generación Renovable**: Predicción de energía eólica diaria con HistGradientBoosting
- **Cobertura**: 87/87 municipios ISTAC en Canarias
- **Datos**: Dataset ampliado con Open-Meteo integrando estaciones meteorológicas reales y datos derivados

## 📊 Resultados Principales

| Modelo | Target | MAE | RMSE | R² | MAPE | WMAPE |
|---|---|---:|---:|---:|---:|---:|
| XGBoost | Consumo Total | 6.48 | 18.23 | 0.9984 | 4.28% | 2.50% |
| XGBoost | Residencial | 1.54 | 4.10 | 0.9994 | 2.87% | 1.69% |
| XGBoost | Servicios | 5.02 | 14.06 | 0.9975 | 5.99% | 3.19% |
| XGBoost | Industria | 1.31 | 5.49 | 0.9700 | 22.25% | 11.45% |
| HistGradientBoosting | Eólica | 653.78 | 853.19 | 0.8985 | 29.82% | 16.77% |

## 🗂️ Estructura del Proyecto

```
tfce-modelos-energia-canarias/
├── tfc-datasets/               # Preparación y gestión de datos
│   ├── inputs/                 # Fuentes originales
│   ├── scripts/                # Scripts de construcción del pipeline
│   ├── outputs/                # Datasets intermedios y finales
│   └── docs/                   # Documentación del pipeline
├── tfc-model/                  # Entrenamiento y evaluación de modelos
│   ├── data/                   # Datasets finales de entrada
│   ├── src/
│   │   ├── training/           # Scripts de entrenamiento
│   │   ├── models/             # Modelos serializados
│   │   └── evaluation/         # Métricas, predicciones y gráficas
│   └── requirements.txt        # Dependencias Python
├── tfc-streamlit/              # Aplicación interactiva
└── README.md
```

## 🔧 Componentes Principales

### 1. **tfc-datasets** - Gestión de Datos

Organiza el pipeline de datos en cuatro bloques:

- **inputs/**: Fuentes originales (ISTAC, REE, estaciones meteorológicas)
- **scripts/**: Construcción y procesamiento de datasets
- **outputs/**: Datasets finales generados
- **docs/**: Documentación del pipeline y variables

**Características principales:**
- Integración de datos de demanda eléctrica (ISTAC)
- Datos de generación renovable (REE)
- Augmentación con Open-Meteo para cobertura completa
- 39 municipios con meteorología observada + 48 con Open-Meteo = 87/87 cobertura

### 2. **tfc-model** - Modelado Predictivo

Entrenamiento y evaluación de modelos con separación temporal:

- **Consumo Eléctrico**: XGBoost con enfoque sectorial
- **Generación Renovable**: HistGradientBoosting para predicción eólica
- Modelos serializados para inferencia
- Métricas y visualizaciones de evaluación

### 3. **tfc-streamlit** - Aplicación Interactiva

Interfaz web para explorar predicciones y datos en tiempo real.

## 📦 Stack Tecnológico

- **Python** (74.9%): XGBoost, scikit-learn, pandas, NumPy
- **JavaScript** (17.6%): Componentes frontend
- **HTML** (7.5%): Templates y interfaz

## 🚀 Instalación Local

### Requisitos previos
- Python 3.8+
- pip o conda

### Configuración

```bash
# Clonar el repositorio
git clone https://github.com/Aimara8/tfce-modelos-energia-canarias.git
cd tfce-modelos-energia-canarias

# Instalar dependencias del modelo
cd tfc-model
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate   # Linux/Mac
pip install -r requirements.txt

# Ejecutar entrenamientos
python src/training/Energy_Consumption/XGBoost.py
python src/training/Renewable_Energy_Generation/HistGradientBoosting_VS_Ridge.py
```

### Ejecutar la aplicación Streamlit localmente

```bash
cd ../tfc-streamlit
pip install streamlit
streamlit run app.py
```

Luego accede a `http://localhost:8501`

## 📚 Documentación

Dentro del proyecto encontrarás:

- **tfc-datasets/docs/**:
  - `Context.md`: Contexto y objetivos
  - `RENEWABLE_DATASET_VARIABLES.md`: Variables de renovables
  - `CONSUMPTION_DATASET_VARIABLES.md`: Variables de consumo
  - `WEATHER_DATASET_VARIABLES.md`: Variables meteorológicas
  - `OPEN_METEO_AUGMENTATION.md`: Detalles de la augmentación

- **tfc-model/**:
  - `README.md`: Estructura y ejecución de modelos
  - Métricas y gráficas en `src/evaluation/`

## 📈 Flujo de Trabajo

```
Datos Originales (ISTAC, REE, Meteorología)
    ↓
tfc-datasets/ (Procesamiento y augmentación)
    ↓
Datasets Finales (87/87 municipios Canarias)
    ↓
tfc-model/ (Entrenamiento y evaluación)
    ↓
Modelos Entrenados + Métricas
    ↓
tfc-streamlit/ (Demostración interactiva)
```

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto es parte del TFCE. Especifica la licencia según corresponda.

## 👤 Autor

**Aimara8**

## 📧 Contacto

Para preguntas o sugerencias, abre un issue en el repositorio.

---

**Última actualización**: 2026-05-26
