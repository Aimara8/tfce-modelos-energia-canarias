from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .predictor import loaded_model_names, predict_consumption, predict_wind, unavailable_models
from .schemas import ConsumptionInput, ConsumptionOutput, HealthResponse, WindInput, WindOutput

app = FastAPI(
    title="API Prediccion Energetica Canarias",
    description="Inferencia de consumo electrico municipal y generacion eolica con los modelos del proyecto.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        modelos_cargados=loaded_model_names(),
        modelos_no_disponibles=unavailable_models(),
    )


@app.post("/predict/consumo", response_model=ConsumptionOutput, tags=["Prediccion"])
def predict_consumption_endpoint(payload: ConsumptionInput) -> ConsumptionOutput:
    result = predict_consumption(payload.model_dump())
    return ConsumptionOutput(
        municipality_enc=payload.municipality_enc,
        fecha=payload.fecha,
        **result,
    )


@app.post("/predict/eolica", response_model=WindOutput, tags=["Prediccion"])
def predict_wind_endpoint(payload: WindInput) -> WindOutput:
    result = predict_wind(payload.model_dump())
    return WindOutput(fecha=payload.fecha, **result)

