"""
main.py — API REST para predicción de demanda eléctrica y generación eólica
en Canarias. Proyecto TFC — Modelado Predictivo de Energía.

Arrancar con:
    uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    ConsumoInput, ConsumoOutput,
    EolicaInput,  EolicaOutput,
    HealthResponse,
)
from api.predictor import predict_consumo, predict_eolica, loaded_model_names

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="API Predicción Energética — Canarias",
    description=(
        "Modelos predictivos de **demanda eléctrica municipal** (XGBoost) "
        "y **generación eólica diaria** (HistGradientBoosting) para el "
        "sistema eléctrico canario.\n\n"
        "Proyecto TFC — CIFP César Manrique."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Permite peticiones desde el cliente Streamlit (localhost u otro origen)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
def health_check():
    """Comprueba que la API está activa y qué modelos están cargados."""
    return HealthResponse(
        status="ok",
        modelos_cargados=loaded_model_names(),
    )


@app.post(
    "/predict/consumo",
    response_model=ConsumoOutput,
    tags=["Predicción"],
    summary="Predicción de demanda eléctrica diaria por municipio",
)
def predict_consumo_endpoint(payload: ConsumoInput):
    """
    Devuelve la predicción de consumo eléctrico diario (MWh) para un
    municipio de Canarias, desagregado en:
    - **total**
    - **residencial**
    - **servicios**
    - **industria**

    Requiere variables meteorológicas y lags del consumo anterior.
    """
    try:
        result = predict_consumo(payload.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ConsumoOutput(
        municipio_cod=payload.municipio_cod,
        fecha=payload.fecha,
        **result,
    )


@app.post(
    "/predict/eolica",
    response_model=EolicaOutput,
    tags=["Predicción"],
    summary="Predicción de generación eólica diaria en Canarias",
)
def predict_eolica_endpoint(payload: EolicaInput):
    """
    Devuelve la predicción de generación eólica diaria agregada (MWh)
    para el sistema eléctrico canario.

    Requiere variables meteorológicas agregadas del archipiélago
    y lags de la generación reciente.
    """
    try:
        result = predict_eolica(payload.model_dump())
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return EolicaOutput(fecha=payload.fecha, **result)
