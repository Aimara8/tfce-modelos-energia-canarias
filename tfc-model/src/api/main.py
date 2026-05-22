from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .predictor import (
    benchmark_api,
    consumption_dashboard,
    loaded_model_names,
    predict_consumption,
    predict_wind,
    project_metadata,
    unavailable_models,
    wind_dashboard,
)
from .schemas import ConsumptionInput, ConsumptionOutput, HealthResponse, MetadataResponse, WindInput, WindOutput

app = FastAPI(
    title="API Prediccion Energetica Canarias",
    description="Inferencia de consumo electrico municipal y generacion eolica con incertidumbre.",
    version="1.1.0",
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


@app.get("/metadata", response_model=MetadataResponse, tags=["Sistema"])
def metadata() -> MetadataResponse:
    return MetadataResponse(**project_metadata())


@app.get("/dashboard/consumo", tags=["Dashboard"])
def dashboard_consumo() -> dict:
    return consumption_dashboard()


@app.get("/dashboard/eolica", tags=["Dashboard"])
def dashboard_eolica() -> dict:
    return wind_dashboard()


@app.get("/benchmark", tags=["Sistema"])
def benchmark() -> dict:
    return benchmark_api()


@app.post("/predict/consumo", response_model=ConsumptionOutput, tags=["Prediccion"])
def predict_consumption_endpoint(payload: ConsumptionInput) -> ConsumptionOutput:
    try:
        result = predict_consumption(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ConsumptionOutput(municipality=payload.municipality, fecha=payload.fecha, **result)


@app.post("/predict/eolica", response_model=WindOutput, tags=["Prediccion"])
def predict_wind_endpoint(payload: WindInput) -> WindOutput:
    try:
        result = predict_wind(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return WindOutput(fecha=payload.fecha, **result)
