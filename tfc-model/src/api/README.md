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
- El selector de municipios sale del dataset oficial, actualmente 87 municipios ISTAC.
- Para fechas futuras, la meteorologia automatica consulta Open-Meteo por coordenadas municipales.
- La prediccion eolica devuelve rango de incertidumbre y nivel de confianza.
- El test final debe quedar fuera del ajuste de hiperparametros y calibraciones.

## Validacion actual

Smoke test ejecutado con el dataset ampliado:

- modelos de consumo: cargados
- modelo eolico: cargado
- municipios con coordenadas configuradas: 87/87
- `canarias_weather_municipality_count` por defecto: 87
