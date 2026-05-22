# API y cliente

## Arrancar API

```powershell
cd tfc-model
python -m uvicorn src.api.main:app --reload --port 8000
```

## Arrancar cliente

```powershell
cd tfc-model
python -m streamlit run src/client/app.py
```

## Enfoque

- Las graficas se generan con codigo en Streamlit y Plotly.
- La API autocompleta historicos cuando hay datos previos suficientes.
- La prediccion eolica devuelve rango de incertidumbre y nivel de confianza.
- El test final debe quedar fuera del ajuste de hiperparametros y calibraciones.
