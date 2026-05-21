# API + Cliente — Predicción Energética Canarias

## Estructura

```
proyecto/
├── models/
│   ├── model_total.joblib
│   ├── model_residencial.joblib
│   ├── model_servicios.joblib
│   ├── model_industria.joblib
│   └── model_eolica.joblib
├── api/
│   ├── __init__.py
│   ├── main.py          ← FastAPI app
│   ├── schemas.py       ← Pydantic (inputs / outputs)
│   └── predictor.py     ← Carga modelos + feature engineering
├── client/
│   └── app.py           ← Streamlit
└── requirements_api.txt
```

## Instalación

```bash
pip install -r requirements_api.txt
```

## Arrancar la API

```bash
uvicorn api.main:app --reload --port 8000
```

- Swagger UI:  http://localhost:8000/docs
- ReDoc:       http://localhost:8000/redoc
- Health:      http://localhost:8000/health

## Arrancar el cliente Streamlit

En otra terminal:

```bash
streamlit run client/app.py
```

Se abre automáticamente en http://localhost:8501

## Endpoints

| Método | Ruta              | Descripción                         |
|--------|-------------------|-------------------------------------|
| GET    | /health           | Estado y modelos cargados           |
| POST   | /predict/consumo  | Demanda eléctrica (4 sectores, MWh) |
| POST   | /predict/eolica   | Generación eólica diaria (MWh)      |

## Notas de adaptación

- En `predictor.py`, revisa que los nombres de columnas en
  `build_consumo_features()` y `build_eolica_features()` coincidan
  **exactamente** con los del entrenamiento (puedes comprobarlos con
  `model.feature_names_in_` si usaste scikit-learn o `model.get_booster().feature_names`
  en XGBoost).
- Si necesitas añadir o quitar features, modifica solo `predictor.py`
  sin tocar `main.py` ni `schemas.py`.
