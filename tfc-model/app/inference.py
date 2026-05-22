from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "src" / "models"

CONSUMPTION_SECTORS = {
    "total": "demand_total_mwh",
    "residencial": "demand_residencial_mwh",
    "servicios": "demand_servicios_mwh",
    "industria": "demand_industria_mwh",
}

CONSUMPTION_FEATURES = [
    "municipality_enc",
    "year",
    "month",
    "dayofweek",
    "dayofyear",
    "quarter",
    "week",
    "is_weekend",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "temp_avg_c",
    "temp_max_c",
    "temp_min_c",
    "temp_range_c",
    "humidity_avg_pct",
    "dew_point_avg_c",
    "pressure_avg_hpa",
    "precip_intensity_avg_mm",
    "rain_daily_mm",
    "wind_speed_avg_ms",
    "wind_speed_max_ms",
    "wind_speed_sdev_ms",
    "wind_dir_avg_deg",
    "weather_station_count",
    "hdd",
    "cdd",
    "lag_1d",
    "lag_7d",
    "lag_14d",
    "lag_28d",
    "rolling_7d_mean",
    "rolling_30d_mean",
    "rolling_7d_std",
]

RENEWABLE_EXCLUDED_COLUMNS = {
    "date",
    "ree_eolica_value",
    "weather_data_source",
    "ree_generacion_renovable_value",
    "ree_eolica_pct",
    "ree_solar_fotovoltaica_pct",
    "ree_hidraulica_pct",
    "ree_hidroeolica_pct",
    "ree_otras_renovables_pct",
    "ree_solar_fotovoltaica_value",
    "ree_hidraulica_value",
    "ree_hidroeolica_value",
    "ree_otras_renovables_value",
}


class ConsumptionPredictionRequest(BaseModel):
    sector: Literal["total", "residencial", "servicios", "industria"]
    municipality: str
    date: date
    temp_avg_c: float
    temp_max_c: float
    temp_min_c: float
    humidity_avg_pct: float
    dew_point_avg_c: float
    pressure_avg_hpa: float
    precip_intensity_avg_mm: float
    rain_daily_mm: float
    wind_speed_avg_ms: float
    wind_speed_max_ms: float
    wind_speed_sdev_ms: float
    wind_dir_avg_deg: float
    weather_station_count: int = Field(ge=0)
    lag_1d: float
    lag_7d: float
    lag_14d: float
    lag_28d: float
    rolling_7d_mean: float
    rolling_30d_mean: float
    rolling_7d_std: float


class RenewablePredictionRequest(BaseModel):
    model: Literal["hgb", "ridge"] = "hgb"
    date: date
    canarias_weather_municipality_count: int = Field(ge=0)
    canarias_weather_station_count: int = Field(ge=0)
    temp_avg_c: float
    temp_max_c: float
    temp_min_c: float
    pressure_avg_hpa: float
    dew_point_avg_c: float
    precip_intensity_avg_mm: float
    rain_daily_mm: float
    humidity_avg_pct: float
    wind_dir_avg_deg: float
    wind_dir_max_deg: float
    wind_dir_sdev_deg: float
    wind_speed_avg_ms: float
    wind_speed_max_ms: float
    wind_speed_sdev_ms: float
    lag_1d: float
    lag_2d: float
    lag_3d: float
    rolling_3d_mean: float
    rolling_7d_mean: float
    rolling_7d_std: float
    rolling_14d_mean: float
    hidroeolica_lag1: float


@lru_cache(maxsize=1)
def get_municipalities() -> list[str]:
    df = pd.read_csv(DATA_DIR / "final_demand_consumption_dataset.csv", usecols=["municipality"])
    return sorted(df["municipality"].dropna().unique().tolist())


@lru_cache(maxsize=1)
def get_consumption_models() -> dict[str, xgb.XGBRegressor]:
    models: dict[str, xgb.XGBRegressor] = {}
    model_root = MODEL_DIR / "consumption_energy_demand"
    for sector in CONSUMPTION_SECTORS:
        model = xgb.XGBRegressor()
        model.load_model(model_root / f"xgboost_{sector}.json")
        models[sector] = model
    return models


@lru_cache(maxsize=1)
def get_renewable_assets() -> dict[str, object]:
    model_root = MODEL_DIR / "renewable_energy_generation"
    return {
        "hgb": joblib.load(model_root / "hgb_eolica.pkl"),
        "ridge": joblib.load(model_root / "ridge_eolica.pkl"),
        "imputer": joblib.load(model_root / "imputer_eolica.pkl"),
    }


@lru_cache(maxsize=1)
def get_renewable_feature_order() -> list[str]:
    df = pd.read_csv(DATA_DIR / "final_renewable_generation_dataset.csv", parse_dates=["date"])
    if "weather_data_source" in df.columns:
        df = df.drop(columns=["weather_data_source"])
    df = df.dropna(subset=["ree_eolica_value"]).sort_values("date").reset_index(drop=True)

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofyear"] = df["date"].dt.dayofyear
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    for col in ["wind_speed_avg_ms", "wind_speed_max_ms", "wind_speed_sdev_ms"]:
        if col in df.columns:
            df[f"{col}_cb"] = df[col] ** 3
            df[f"{col}_sq"] = df[col] ** 2

    if "wind_speed_avg_ms" in df.columns and "wind_dir_sdev_deg" in df.columns:
        df["wind_chaos"] = df["wind_speed_avg_ms"] * df["wind_dir_sdev_deg"]

    if "wind_speed_avg_ms" in df.columns and "wind_speed_sdev_ms" in df.columns:
        df["wind_variability_ratio"] = df["wind_speed_sdev_ms"] / (df["wind_speed_avg_ms"] + 0.1)

    if "wind_speed_avg_ms" in df.columns:
        df["wind_x_month_sin"] = df["wind_speed_avg_ms"] * df["month_sin"]
        df["wind_x_month_cos"] = df["wind_speed_avg_ms"] * df["month_cos"]
        df["wind_cb_x_month_sin"] = df["wind_speed_avg_ms_cb"] * df["month_sin"]

    for lag in [1, 2, 3]:
        df[f"lag_{lag}d"] = df["ree_eolica_value"].shift(lag)

    df["rolling_3d_mean"] = df["ree_eolica_value"].shift(1).rolling(3, min_periods=2).mean()
    df["rolling_7d_mean"] = df["ree_eolica_value"].shift(1).rolling(7, min_periods=3).mean()
    df["rolling_7d_std"] = df["ree_eolica_value"].shift(1).rolling(7, min_periods=3).std()
    df["rolling_14d_mean"] = df["ree_eolica_value"].shift(1).rolling(14, min_periods=5).mean()
    df["hidroeolica_lag1"] = df["ree_hidroeolica_value"].shift(1)

    return [
        col
        for col in df.columns
        if col not in RENEWABLE_EXCLUDED_COLUMNS
        and pd.api.types.is_numeric_dtype(df[col])
    ]


def build_consumption_features(payload: ConsumptionPredictionRequest) -> pd.DataFrame:
    municipalities = get_municipalities()
    if payload.municipality not in municipalities:
        raise ValueError(f"Municipio no reconocido: {payload.municipality}")

    ts = pd.Timestamp(payload.date)
    month = int(ts.month)
    dayofweek = int(ts.dayofweek)

    row = {
        "municipality_enc": municipalities.index(payload.municipality),
        "year": int(ts.year),
        "month": month,
        "dayofweek": dayofweek,
        "dayofyear": int(ts.dayofyear),
        "quarter": int(ts.quarter),
        "week": int(ts.isocalendar().week),
        "is_weekend": int(dayofweek >= 5),
        "month_sin": float(np.sin(2 * np.pi * month / 12)),
        "month_cos": float(np.cos(2 * np.pi * month / 12)),
        "dow_sin": float(np.sin(2 * np.pi * dayofweek / 7)),
        "dow_cos": float(np.cos(2 * np.pi * dayofweek / 7)),
        "temp_avg_c": payload.temp_avg_c,
        "temp_max_c": payload.temp_max_c,
        "temp_min_c": payload.temp_min_c,
        "temp_range_c": payload.temp_max_c - payload.temp_min_c,
        "humidity_avg_pct": payload.humidity_avg_pct,
        "dew_point_avg_c": payload.dew_point_avg_c,
        "pressure_avg_hpa": payload.pressure_avg_hpa,
        "precip_intensity_avg_mm": payload.precip_intensity_avg_mm,
        "rain_daily_mm": payload.rain_daily_mm,
        "wind_speed_avg_ms": payload.wind_speed_avg_ms,
        "wind_speed_max_ms": payload.wind_speed_max_ms,
        "wind_speed_sdev_ms": payload.wind_speed_sdev_ms,
        "wind_dir_avg_deg": payload.wind_dir_avg_deg,
        "weather_station_count": payload.weather_station_count,
        "hdd": float(max(18.0 - payload.temp_avg_c, 0.0)),
        "cdd": float(max(payload.temp_avg_c - 18.0, 0.0)),
        "lag_1d": payload.lag_1d,
        "lag_7d": payload.lag_7d,
        "lag_14d": payload.lag_14d,
        "lag_28d": payload.lag_28d,
        "rolling_7d_mean": payload.rolling_7d_mean,
        "rolling_30d_mean": payload.rolling_30d_mean,
        "rolling_7d_std": payload.rolling_7d_std,
    }
    return pd.DataFrame([[row[col] for col in CONSUMPTION_FEATURES]], columns=CONSUMPTION_FEATURES)


def build_renewable_features(payload: RenewablePredictionRequest) -> pd.DataFrame:
    ts = pd.Timestamp(payload.date)
    month = int(ts.month)
    dayofweek = int(ts.dayofweek)
    dayofyear = int(ts.dayofyear)
    row = {
        "canarias_weather_municipality_count": payload.canarias_weather_municipality_count,
        "canarias_weather_station_count": payload.canarias_weather_station_count,
        "temp_avg_c": payload.temp_avg_c,
        "temp_max_c": payload.temp_max_c,
        "temp_min_c": payload.temp_min_c,
        "pressure_avg_hpa": payload.pressure_avg_hpa,
        "dew_point_avg_c": payload.dew_point_avg_c,
        "precip_intensity_avg_mm": payload.precip_intensity_avg_mm,
        "rain_daily_mm": payload.rain_daily_mm,
        "humidity_avg_pct": payload.humidity_avg_pct,
        "wind_dir_avg_deg": payload.wind_dir_avg_deg,
        "wind_dir_max_deg": payload.wind_dir_max_deg,
        "wind_dir_sdev_deg": payload.wind_dir_sdev_deg,
        "wind_speed_avg_ms": payload.wind_speed_avg_ms,
        "wind_speed_max_ms": payload.wind_speed_max_ms,
        "wind_speed_sdev_ms": payload.wind_speed_sdev_ms,
        "year": int(ts.year),
        "month": month,
        "dayofweek": dayofweek,
        "dayofyear": dayofyear,
        "quarter": int(ts.quarter),
        "is_weekend": int(dayofweek >= 5),
        "month_sin": float(np.sin(2 * np.pi * month / 12)),
        "month_cos": float(np.cos(2 * np.pi * month / 12)),
        "doy_sin": float(np.sin(2 * np.pi * dayofyear / 365.25)),
        "doy_cos": float(np.cos(2 * np.pi * dayofyear / 365.25)),
        "dow_sin": float(np.sin(2 * np.pi * dayofweek / 7)),
        "dow_cos": float(np.cos(2 * np.pi * dayofweek / 7)),
        "wind_speed_avg_ms_cb": payload.wind_speed_avg_ms ** 3,
        "wind_speed_avg_ms_sq": payload.wind_speed_avg_ms ** 2,
        "wind_speed_max_ms_cb": payload.wind_speed_max_ms ** 3,
        "wind_speed_max_ms_sq": payload.wind_speed_max_ms ** 2,
        "wind_speed_sdev_ms_cb": payload.wind_speed_sdev_ms ** 3,
        "wind_speed_sdev_ms_sq": payload.wind_speed_sdev_ms ** 2,
        "wind_chaos": payload.wind_speed_avg_ms * payload.wind_dir_sdev_deg,
        "wind_variability_ratio": payload.wind_speed_sdev_ms / (payload.wind_speed_avg_ms + 0.1),
        "wind_x_month_sin": payload.wind_speed_avg_ms * np.sin(2 * np.pi * month / 12),
        "wind_x_month_cos": payload.wind_speed_avg_ms * np.cos(2 * np.pi * month / 12),
        "wind_cb_x_month_sin": (payload.wind_speed_avg_ms ** 3) * np.sin(2 * np.pi * month / 12),
        "lag_1d": payload.lag_1d,
        "lag_2d": payload.lag_2d,
        "lag_3d": payload.lag_3d,
        "rolling_3d_mean": payload.rolling_3d_mean,
        "rolling_7d_mean": payload.rolling_7d_mean,
        "rolling_7d_std": payload.rolling_7d_std,
        "rolling_14d_mean": payload.rolling_14d_mean,
        "hidroeolica_lag1": payload.hidroeolica_lag1,
    }

    feature_order = get_renewable_feature_order()
    return pd.DataFrame([[row[col] for col in feature_order]], columns=feature_order)


def predict_consumption(payload: ConsumptionPredictionRequest) -> dict[str, object]:
    features = build_consumption_features(payload)
    model = get_consumption_models()[payload.sector]
    prediction = float(model.predict(features)[0])
    return {
        "sector": payload.sector,
        "municipality": payload.municipality,
        "date": payload.date.isoformat(),
        "prediction_mwh": prediction,
    }


def predict_renewable(payload: RenewablePredictionRequest) -> dict[str, object]:
    features = build_renewable_features(payload)
    assets = get_renewable_assets()
    if payload.model == "hgb":
        transformed = assets["imputer"].transform(features.values)
        prediction = float(assets["hgb"].predict(transformed)[0])
    else:
        prediction = float(assets["ridge"].predict(features)[0])

    return {
        "model": payload.model,
        "target": "ree_eolica_value",
        "date": payload.date.isoformat(),
        "prediction_mwh": prediction,
    }
