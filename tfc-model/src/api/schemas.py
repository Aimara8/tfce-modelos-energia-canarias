from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class WeatherInput(BaseModel):
    temp_avg_c: float = Field(..., description="Temperatura media diaria en grados Celsius")
    temp_max_c: float = Field(..., description="Temperatura maxima diaria en grados Celsius")
    temp_min_c: float = Field(..., description="Temperatura minima diaria en grados Celsius")
    humidity_avg_pct: float | None = Field(None, ge=0, le=100)
    dew_point_avg_c: float | None = None
    pressure_avg_hpa: float | None = None
    precip_intensity_avg_mm: float = Field(0.0, ge=0)
    rain_daily_mm: float = Field(0.0, ge=0)
    wind_speed_avg_ms: float | None = Field(None, ge=0)
    wind_speed_max_ms: float | None = Field(None, ge=0)
    wind_speed_sdev_ms: float | None = Field(None, ge=0)
    wind_dir_avg_deg: float | None = Field(None, ge=0, le=360)
    wind_dir_max_deg: float | None = Field(None, ge=0, le=360)
    wind_dir_sdev_deg: float | None = Field(None, ge=0, le=180)
    weather_station_count: int = Field(1, ge=1)


class ConsumptionHistoryInput(BaseModel):
    lag_1d: float
    lag_7d: float
    lag_14d: float
    lag_28d: float
    rolling_7d_mean: float
    rolling_30d_mean: float
    rolling_7d_std: float


class ConsumptionInput(BaseModel):
    municipality: str
    fecha: date
    weather: WeatherInput | None = None
    history: ConsumptionHistoryInput | None = None


class WindHistoryInput(BaseModel):
    lag_1d: float
    lag_2d: float
    lag_3d: float
    rolling_3d_mean: float
    rolling_7d_mean: float
    rolling_7d_std: float
    rolling_14d_mean: float
    hidroeolica_lag1: float | None = None


class WindInput(BaseModel):
    fecha: date
    weather: WeatherInput | None = None
    history: WindHistoryInput | None = None
    canarias_weather_municipality_count: int = Field(87, ge=1)
    canarias_weather_station_count: int = Field(50, ge=1)


class ChartPoint(BaseModel):
    label: str
    value: float | None


class SectorPrediction(BaseModel):
    sector: Literal["total", "residencial", "servicios", "industria"]
    mwh: float | None
    baseline_mwh: float | None
    history_source: Literal["historical", "manual"]


class ConsumptionOutput(BaseModel):
    municipality: str
    municipality_enc: int
    fecha: date
    predictions: list[SectorPrediction]
    chart_bars: list[ChartPoint]
    chart_reference: list[ChartPoint]
    warnings: list[str]
    model_status: dict[str, str]


class WindOutput(BaseModel):
    fecha: date
    eolica_predicha_mwh: float | None
    uncertainty_low_mwh: float | None
    uncertainty_high_mwh: float | None
    confidence: Literal["alta", "media", "baja", "sin_modelo"]
    condition: Literal["baja", "normal", "favorable", "sin_modelo"]
    comparison_to_rolling_7d_pct: float | None
    chart_series: list[ChartPoint]
    sensitivity_by_wind: list[ChartPoint]
    warnings: list[str]
    model_status: dict[str, str]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    modelos_cargados: list[str]
    modelos_no_disponibles: dict[str, str]
    version: str = "1.1.0"


class MetadataResponse(BaseModel):
    consumption: dict
    renewable: dict
    evaluation: dict
