"""
predictor.py — Carga los modelos serializados y construye las features
con los nombres EXACTOS usados en el entrenamiento.
"""

import math
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
_models: dict = {}

FEATURES_CONSUMO = [
    "municipality_enc",
    "year", "month", "dayofweek", "dayofyear", "quarter", "week", "is_weekend",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "temp_avg_c", "temp_max_c", "temp_min_c", "temp_range_c",
    "humidity_avg_pct", "dew_point_avg_c",
    "pressure_avg_hpa",
    "precip_intensity_avg_mm", "rain_daily_mm",
    "wind_speed_avg_ms", "wind_speed_max_ms", "wind_speed_sdev_ms",
    "wind_dir_avg_deg",
    "weather_station_count",
    "hdd", "cdd",
    "lag_1d", "lag_7d", "lag_14d", "lag_28d",
    "rolling_7d_mean", "rolling_30d_mean", "rolling_7d_std",
]


def _load_models() -> dict:
    loaded = {}
    for name, filename in {
        "total":       "model_total.joblib",
        "residencial": "model_residencial.joblib",
        "servicios":   "model_servicios.joblib",
        "industria":   "model_industria.joblib",
        "eolica":      "model_eolica.joblib",
    }.items():
        path = MODELS_DIR / filename
        if path.exists():
            loaded[name] = joblib.load(path)
        else:
            print(f"[WARN] Modelo no encontrado: {path}")
    return loaded


def get_models() -> dict:
    global _models
    if not _models:
        _models = _load_models()
    return _models


def _cyclic(value: float, period: float):
    angle = 2 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


def _hdd_cdd(temp_avg: float, base: float = 15.0):
    return max(0.0, base - temp_avg), max(0.0, temp_avg - base)


def _dew_point(temp_avg: float, humidity: float) -> float:
    """Aproximación Magnus si el cliente no la envía."""
    if humidity <= 0:
        return temp_avg - 5.0
    import math
    a, b = 17.27, 237.7
    gamma = (a * temp_avg / (b + temp_avg)) + math.log(humidity / 100.0)
    return (b * gamma) / (a - gamma)


# ── Feature engineering consumo ──────────────────────────────────────────────
def build_consumo_features(data: dict) -> pd.DataFrame:
    fecha = datetime.strptime(data["fecha"], "%Y-%m-%d")

    month       = fecha.month
    dayofweek   = fecha.weekday()
    dayofyear   = fecha.timetuple().tm_yday
    year        = fecha.year
    quarter     = (month - 1) // 3 + 1
    week        = fecha.isocalendar()[1]
    is_weekend  = int(dayofweek >= 5)

    month_sin, month_cos = _cyclic(month,     12)
    dow_sin,   dow_cos   = _cyclic(dayofweek,  7)

    temp_avg  = data["temp_avg_c"]
    temp_max  = data["temp_max_c"]
    temp_min  = data["temp_min_c"]
    temp_range = temp_max - temp_min
    hdd, cdd  = _hdd_cdd(temp_avg)

    humidity  = data.get("humidity_avg_pct") or np.nan
    dew_point = data.get("dew_point_avg_c")
    if dew_point is None and not np.isnan(humidity):
        dew_point = _dew_point(temp_avg, humidity)

    row = {
        "municipality_enc":       data["municipality_enc"],
        "year":                   year,
        "month":                  month,
        "dayofweek":              dayofweek,
        "dayofyear":              dayofyear,
        "quarter":                quarter,
        "week":                   week,
        "is_weekend":             is_weekend,
        "month_sin":              month_sin,
        "month_cos":              month_cos,
        "dow_sin":                dow_sin,
        "dow_cos":                dow_cos,
        "temp_avg_c":             temp_avg,
        "temp_max_c":             temp_max,
        "temp_min_c":             temp_min,
        "temp_range_c":           temp_range,
        "humidity_avg_pct":       humidity,
        "dew_point_avg_c":        dew_point if dew_point is not None else np.nan,
        "pressure_avg_hpa":       data.get("pressure_avg_hpa") or np.nan,
        "precip_intensity_avg_mm": data.get("precip_intensity_avg_mm") or 0.0,
        "rain_daily_mm":          data.get("rain_daily_mm") or 0.0,
        "wind_speed_avg_ms":      data.get("wind_speed_avg_ms") or np.nan,
        "wind_speed_max_ms":      data.get("wind_speed_max_ms") or np.nan,
        "wind_speed_sdev_ms":     data.get("wind_speed_sdev_ms") or np.nan,
        "wind_dir_avg_deg":       data.get("wind_dir_avg_deg") or np.nan,
        "weather_station_count":  data.get("weather_station_count") or 1,
        "hdd":                    hdd,
        "cdd":                    cdd,
        "lag_1d":                 data["lag_1d"],
        "lag_7d":                 data["lag_7d"],
        "lag_14d":                data["lag_14d"],
        "lag_28d":                data["lag_28d"],
        "rolling_7d_mean":        data["rolling_7d_mean"],
        "rolling_30d_mean":       data["rolling_30d_mean"],
        "rolling_7d_std":         data["rolling_7d_std"],
    }

    df = pd.DataFrame([row])
    # Garantiza el orden exacto del entrenamiento
    return df[FEATURES_CONSUMO]


# ── Feature engineering eólico ────────────────────────────────────────────────
def build_eolica_features(data: dict) -> pd.DataFrame:
    fecha = datetime.strptime(data["fecha"], "%Y-%m-%d")
    month       = fecha.month
    dayofweek   = fecha.weekday()
    dayofyear   = fecha.timetuple().tm_yday

    month_sin, month_cos = _cyclic(month,     12)
    doy_sin,   doy_cos   = _cyclic(dayofyear, 365)

    wind     = data["wind_speed_avg_ms"]
    wind_max = data["wind_speed_max_ms"]
    wind_std = data.get("wind_speed_sdev_ms") or np.nan
    wind2    = wind ** 2
    wind3    = wind ** 3

    wind_chaos             = wind_std * wind if not np.isnan(wind_std) else np.nan
    wind_variability_ratio = (wind_std / wind) if (wind > 0 and not np.isnan(wind_std)) else np.nan

    row = {
        "month":                  month,
        "dayofweek":              dayofweek,
        "dayofyear":              dayofyear,
        "month_sin":              month_sin,
        "month_cos":              month_cos,
        "doy_sin":                doy_sin,
        "doy_cos":                doy_cos,
        "wind_speed_avg_ms":      wind,
        "wind_speed_max_ms":      wind_max,
        "wind_speed_sdev_ms":     wind_std,
        "wind_speed2":            wind2,
        "wind_speed3":            wind3,
        "wind_chaos":             wind_chaos,
        "wind_variability_ratio": wind_variability_ratio,
        "temp_avg_c":             data.get("temp_avg_c") or np.nan,
        "humidity_avg_pct":       data.get("humidity_avg_pct") or np.nan,
        "eolica_lag1":            data["eolica_lag1"],
        "eolica_lag2":            data["eolica_lag2"],
        "eolica_lag3":            data["eolica_lag3"],
        "eolica_rolling3":        data["eolica_rolling3"],
    }
    return pd.DataFrame([row])


# ── Predicción ────────────────────────────────────────────────────────────────
def predict_consumo(data: dict) -> dict:
    models   = get_models()
    features = build_consumo_features(data)
    result   = {}
    for sector in ["total", "residencial", "servicios", "industria"]:
        if sector in models:
            pred = float(models[sector].predict(features)[0])
            result[f"demand_{sector}_mwh"] = round(max(pred, 0.0), 4)
        else:
            result[f"demand_{sector}_mwh"] = None
    return result


def predict_eolica(data: dict) -> dict:
    models = get_models()
    if "eolica" not in models:
        raise RuntimeError("Modelo eólico no disponible")
    features = build_eolica_features(data)
    pred = float(models["eolica"].predict(features)[0])
    return {"eolica_predicha_mwh": round(max(pred, 0.0), 4)}


def loaded_model_names() -> list[str]:
    return list(get_models().keys())
