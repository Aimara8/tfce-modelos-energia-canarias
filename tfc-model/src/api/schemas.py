from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class WeatherInput(BaseModel):
    temp_avg_c: float = Field(..., description="Temperatura media diaria en grados Celsius")
    temp_max_c: float = Field(..., description="Temperatura maxima diaria en grados Celsius")
    temp_min_c: float = Field(..., description="Temperatura minima diaria en grados Celsius")
    humidity_avg_pct: float | None = Field(None, description="Humedad relativa media")
    dew_point_avg_c: float | None = Field(None, description="Punto de rocio medio")
    pressure_avg_hpa: float | None = Field(None, description="Presion atmosferica media")
    precip_intensity_avg_mm: float = Field(0.0, description="Intensidad media de precipitacion")
    rain_daily_mm: float = Field(0.0, description="Lluvia diaria acumulada")
    wind_speed_avg_ms: float | None = Field(None, description="Velocidad media del viento")
    wind_speed_max_ms: float | None = Field(None, description="Velocidad maxima del viento")
    wind_speed_sdev_ms: float | None = Field(None, description="Desviacion estandar de velocidad del viento")
    wind_dir_avg_deg: float | None = Field(None, description="Direccion media del viento")
    wind_dir_max_deg: float | None = Field(None, description="Direccion maxima del viento")
    wind_dir_sdev_deg: float | None = Field(None, description="Variabilidad de direccion del viento")
    weather_station_count: int = Field(1, ge=1, description="Numero de estaciones usadas")


class ConsumptionHistoryInput(BaseModel):
    lag_1d: float
    lag_7d: float
    lag_14d: float
    lag_28d: float
    rolling_7d_mean: float
    rolling_30d_mean: float
    rolling_7d_std: float


class ConsumptionInput(BaseModel):
    municipality_enc: int = Field(..., ge=0, description="Codigo interno del LabelEncoder del entrenamiento")
    fecha: date
    weather: WeatherInput
    history: ConsumptionHistoryInput


class SectorPrediction(BaseModel):
    sector: Literal["total", "residencial", "servicios", "industria"]
    mwh: float | None


class ChartPoint(BaseModel):
    label: str
    value: float | None


class ConsumptionOutput(BaseModel):
    municipality_enc: int
    fecha: date
    predictions: list[SectorPrediction]
    chart_bars: list[ChartPoint]
    chart_reference: list[ChartPoint]
    model_status: dict[str, str]


class WindHistoryInput(BaseModel):
    lag_1d: float = Field(..., description="Generacion eolica de ayer")
    lag_2d: float = Field(..., description="Generacion eolica de hace 2 dias")
    lag_3d: float = Field(..., description="Generacion eolica de hace 3 dias")
    rolling_3d_mean: float
    rolling_7d_mean: float
    rolling_7d_std: float
    rolling_14d_mean: float
    hidroeolica_lag1: float | None = Field(None, description="Generacion hidroeolica del dia anterior")


class WindInput(BaseModel):
    fecha: date
    weather: WeatherInput
    history: WindHistoryInput
    canarias_weather_municipality_count: int = Field(80, ge=1)
    canarias_weather_station_count: int = Field(100, ge=1)


class WindOutput(BaseModel):
    fecha: date
    eolica_predicha_mwh: float | None
    condition: Literal["baja", "normal", "favorable", "sin_modelo"]
    chart_series: list[ChartPoint]
    sensitivity_by_wind: list[ChartPoint]
    model_status: dict[str, str]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    modelos_cargados: list[str]
    modelos_no_disponibles: dict[str, str]
    version: str = "1.0.0"

