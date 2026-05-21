# API y cliente

Esta API usa los artefactos reales guardados en `tfc-model/src/models`.

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

## Prueba rapida sin servidor

```powershell
cd tfc-model
python src/api/smoke_test.py
```

## Notas

- El cliente ya no pide todas las features tecnicas. La API calcula calendario, variables ciclicas, `hdd`, `cdd`, rangos e interacciones.
- El consumo sigue necesitando `municipality_enc` porque el `LabelEncoder` original no esta persistido en el repo.
- La reduccion dimensional de eolica debe evaluarse con `data/final_renewable_generation_dataset.csv`. Si el CSV no esta presente, primero hay que regenerarlo desde `tfc-datasets`.
- Los modelos eolicos `.pkl` fueron serializados con `scikit-learn 1.8.0`; conviene alinear esa version antes de presentar resultados finales.
