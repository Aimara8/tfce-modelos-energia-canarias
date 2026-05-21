from pydantic import BaseModel, Field
from typing import Optional


class ConsumoInput(BaseModel):
    # Identificación
    municipality_enc: int = Field(..., description="Código numérico del municipio (label encoding del entrenamiento)")
    fecha: str           = Field(..., description="Fecha YYYY-MM-DD")

    # Meteorología — nombres exactos del dataset
    temp_avg_c:               float           = Field(...,  description="Temperatura media (°C)")
    temp_max_c:               float           = Field(...,  description="Temperatura máxima (°C)")
    temp_min_c:               float           = Field(...,  description="Temperatura mínima (°C)")
    humidity_avg_pct:         Optional[float] = Field(None, description="Humedad relativa media (%)")
    dew_point_avg_c:          Optional[float] = Field(None, description="Punto de rocío (°C) — se calcula si no se envía")
    pressure_avg_hpa:         Optional[float] = Field(None, description="Presión media (hPa)")
    precip_intensity_avg_mm:  Optional[float] = Field(0.0,  description="Intensidad precipitación media (mm)")
    rain_daily_mm:            Optional[float] = Field(0.0,  description="Precipitación diaria total (mm)")
    wind_speed_avg_ms:        Optional[float] = Field(None, description="Velocidad media viento (m/s)")
    wind_speed_max_ms:        Optional[float] = Field(None, description="Velocidad máxima viento (m/s)")
    wind_speed_sdev_ms:       Optional[float] = Field(None, description="Desviación estándar velocidad viento")
    wind_dir_avg_deg:         Optional[float] = Field(None, description="Dirección media del viento (grados)")
    weather_station_count:    Optional[int]   = Field(1,    description="Número de estaciones meteorológicas que aportaron datos")

    # Lags — nombres exactos del entrenamiento
    lag_1d:          float = Field(..., description="Consumo target hace 1 día (MWh)")
    lag_7d:          float = Field(..., description="Consumo target hace 7 días (MWh)")
    lag_14d:         float = Field(..., description="Consumo target hace 14 días (MWh)")
    lag_28d:         float = Field(..., description="Consumo target hace 28 días (MWh)")
    rolling_7d_mean: float = Field(..., description="Media móvil 7 días del target (MWh)")
    rolling_30d_mean:float = Field(..., description="Media móvil 30 días del target (MWh)")
    rolling_7d_std:  float = Field(..., description="Desviación estándar móvil 7 días del target")

    class Config:
        json_schema_extra = {
            "example": {
                "municipality_enc": 12,
                "fecha": "2024-06-15",
                "temp_avg_c": 23.5, "temp_max_c": 28.0, "temp_min_c": 19.0,
                "humidity_avg_pct": 65.0, "pressure_avg_hpa": 1015.0,
                "rain_daily_mm": 0.0,
                "wind_speed_avg_ms": 3.2, "wind_speed_max_ms": 6.1, "wind_speed_sdev_ms": 1.1,
                "wind_dir_avg_deg": 45.0, "weather_station_count": 3,
                "lag_1d": 450.2, "lag_7d": 443.1, "lag_14d": 448.7, "lag_28d": 441.3,
                "rolling_7d_mean": 446.5, "rolling_30d_mean": 444.0, "rolling_7d_std": 8.2,
            }
        }


class ConsumoOutput(BaseModel):
    municipality_enc: int
    fecha: str
    demand_total_mwh:       Optional[float]
    demand_residencial_mwh: Optional[float]
    demand_servicios_mwh:   Optional[float]
    demand_industria_mwh:   Optional[float]
    modelo: str = "XGBoost"


class EolicaInput(BaseModel):
    fecha: str = Field(..., description="Fecha YYYY-MM-DD")
    wind_speed_avg_ms:   float           = Field(...,  description="Velocidad media viento archipiélago (m/s)")
    wind_speed_max_ms:   float           = Field(...,  description="Velocidad máxima viento (m/s)")
    wind_speed_sdev_ms:  Optional[float] = Field(None, description="Desviación estándar velocidad viento")
    temp_avg_c:          Optional[float] = Field(None, description="Temperatura media (°C)")
    humidity_avg_pct:    Optional[float] = Field(None, description="Humedad media (%)")
    eolica_lag1:         float           = Field(...,  description="Generación eólica día anterior (MWh)")
    eolica_lag2:         float           = Field(...,  description="Generación eólica hace 2 días (MWh)")
    eolica_lag3:         float           = Field(...,  description="Generación eólica hace 3 días (MWh)")
    eolica_rolling3:     float           = Field(...,  description="Media móvil 3 días generación eólica (MWh)")

    class Config:
        json_schema_extra = {
            "example": {
                "fecha": "2024-06-15",
                "wind_speed_avg_ms": 7.4, "wind_speed_max_ms": 12.1, "wind_speed_sdev_ms": 2.3,
                "temp_avg_c": 22.0, "humidity_avg_pct": 70.0,
                "eolica_lag1": 3850.0, "eolica_lag2": 3720.0,
                "eolica_lag3": 4010.0, "eolica_rolling3": 3860.0,
            }
        }


class EolicaOutput(BaseModel):
    fecha: str
    eolica_predicha_mwh: float
    modelo: str = "HistGradientBoosting"


class HealthResponse(BaseModel):
    status: str
    modelos_cargados: list[str]
    version: str = "1.0.0"
