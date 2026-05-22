from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

BASE_TEMP_CONSUMPTION = 18.0

CONSUMPTION_TARGETS = {
    "total": "demand_total_mwh",
    "residencial": "demand_residencial_mwh",
    "servicios": "demand_servicios_mwh",
    "industria": "demand_industria_mwh",
}

CONSUMPTION_FEATURES = [
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

WIND_FEATURES = [
    "canarias_weather_municipality_count",
    "canarias_weather_station_count",
    "temp_avg_c", "temp_max_c", "temp_min_c", "pressure_avg_hpa",
    "dew_point_avg_c", "precip_intensity_avg_mm", "rain_daily_mm",
    "humidity_avg_pct", "wind_dir_avg_deg", "wind_dir_max_deg", "wind_dir_sdev_deg",
    "wind_speed_avg_ms", "wind_speed_max_ms", "wind_speed_sdev_ms",
    "year", "month", "dayofweek", "dayofyear", "quarter", "is_weekend",
    "month_sin", "month_cos", "doy_sin", "doy_cos", "dow_sin", "dow_cos",
    "wind_speed_avg_ms_cb", "wind_speed_avg_ms_sq",
    "wind_speed_max_ms_cb", "wind_speed_max_ms_sq",
    "wind_speed_sdev_ms_cb", "wind_speed_sdev_ms_sq",
    "wind_chaos", "wind_variability_ratio",
    "wind_x_month_sin", "wind_x_month_cos", "wind_cb_x_month_sin",
    "lag_1d", "lag_2d", "lag_3d",
    "rolling_3d_mean", "rolling_7d_mean", "rolling_7d_std", "rolling_14d_mean",
    "hidroeolica_lag1",
]


def _as_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def _cyclic(value: float, period: float) -> tuple[float, float]:
    angle = 2 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


def _dew_point(temp_avg: float, humidity: float | None) -> float:
    if humidity is None or humidity <= 0:
        return float("nan")
    a, b = 17.27, 237.7
    gamma = (a * temp_avg / (b + temp_avg)) + math.log(humidity / 100.0)
    return (b * gamma) / (a - gamma)


def _date_features(value: str | date, include_week: bool) -> dict[str, Any]:
    parsed = _as_date(value)
    month_sin, month_cos = _cyclic(parsed.month, 12)
    dow_sin, dow_cos = _cyclic(parsed.weekday(), 7)
    features: dict[str, Any] = {
        "year": parsed.year,
        "month": parsed.month,
        "dayofweek": parsed.weekday(),
        "dayofyear": parsed.timetuple().tm_yday,
        "quarter": (parsed.month - 1) // 3 + 1,
        "is_weekend": int(parsed.weekday() >= 5),
        "month_sin": month_sin,
        "month_cos": month_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
    }
    if include_week:
        features["week"] = parsed.isocalendar()[1]
    return features


def build_consumption_features(
    *,
    fecha: str | date,
    municipality_enc: int,
    weather: dict[str, Any],
    history: dict[str, Any],
) -> pd.DataFrame:
    temp_avg = weather["temp_avg_c"]
    humidity = weather.get("humidity_avg_pct")
    dew_point = weather.get("dew_point_avg_c")
    if dew_point is None:
        dew_point = _dew_point(temp_avg, humidity)

    row = {
        "municipality_enc": municipality_enc,
        **_date_features(fecha, include_week=True),
        "temp_avg_c": temp_avg,
        "temp_max_c": weather["temp_max_c"],
        "temp_min_c": weather["temp_min_c"],
        "temp_range_c": weather["temp_max_c"] - weather["temp_min_c"],
        "humidity_avg_pct": humidity,
        "dew_point_avg_c": dew_point,
        "pressure_avg_hpa": weather.get("pressure_avg_hpa"),
        "precip_intensity_avg_mm": weather.get("precip_intensity_avg_mm", 0.0),
        "rain_daily_mm": weather.get("rain_daily_mm", 0.0),
        "wind_speed_avg_ms": weather.get("wind_speed_avg_ms"),
        "wind_speed_max_ms": weather.get("wind_speed_max_ms"),
        "wind_speed_sdev_ms": weather.get("wind_speed_sdev_ms"),
        "wind_dir_avg_deg": weather.get("wind_dir_avg_deg"),
        "weather_station_count": weather.get("weather_station_count", 1),
        "hdd": max(BASE_TEMP_CONSUMPTION - temp_avg, 0.0),
        "cdd": max(temp_avg - BASE_TEMP_CONSUMPTION, 0.0),
        **history,
    }
    return pd.DataFrame([row], columns=CONSUMPTION_FEATURES)


def build_wind_features(payload: dict[str, Any]) -> pd.DataFrame:
    weather = payload["weather"]
    history = payload["history"]
    date_features = _date_features(payload["fecha"], include_week=False)
    doy_sin, doy_cos = _cyclic(date_features["dayofyear"], 365.25)
    wind_avg = weather.get("wind_speed_avg_ms")
    wind_max = weather.get("wind_speed_max_ms")
    wind_std = weather.get("wind_speed_sdev_ms")
    wind_dir_std = weather.get("wind_dir_sdev_deg")

    row = {
        "canarias_weather_municipality_count": payload.get("canarias_weather_municipality_count"),
        "canarias_weather_station_count": payload.get("canarias_weather_station_count"),
        "temp_avg_c": weather.get("temp_avg_c"),
        "temp_max_c": weather.get("temp_max_c"),
        "temp_min_c": weather.get("temp_min_c"),
        "pressure_avg_hpa": weather.get("pressure_avg_hpa"),
        "dew_point_avg_c": weather.get("dew_point_avg_c") or _dew_point(weather["temp_avg_c"], weather.get("humidity_avg_pct")),
        "precip_intensity_avg_mm": weather.get("precip_intensity_avg_mm", 0.0),
        "rain_daily_mm": weather.get("rain_daily_mm", 0.0),
        "humidity_avg_pct": weather.get("humidity_avg_pct"),
        "wind_dir_avg_deg": weather.get("wind_dir_avg_deg"),
        "wind_dir_max_deg": weather.get("wind_dir_max_deg"),
        "wind_dir_sdev_deg": wind_dir_std,
        "wind_speed_avg_ms": wind_avg,
        "wind_speed_max_ms": wind_max,
        "wind_speed_sdev_ms": wind_std,
        **date_features,
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,
        "wind_speed_avg_ms_cb": _power(wind_avg, 3),
        "wind_speed_avg_ms_sq": _power(wind_avg, 2),
        "wind_speed_max_ms_cb": _power(wind_max, 3),
        "wind_speed_max_ms_sq": _power(wind_max, 2),
        "wind_speed_sdev_ms_cb": _power(wind_std, 3),
        "wind_speed_sdev_ms_sq": _power(wind_std, 2),
        "wind_chaos": _multiply(wind_avg, wind_dir_std),
        "wind_variability_ratio": _ratio(wind_std, (wind_avg or 0) + 0.1),
        "wind_x_month_sin": _multiply(wind_avg, date_features["month_sin"]),
        "wind_x_month_cos": _multiply(wind_avg, date_features["month_cos"]),
        "wind_cb_x_month_sin": _multiply(_power(wind_avg, 3), date_features["month_sin"]),
        **history,
    }
    return pd.DataFrame([row], columns=WIND_FEATURES)


def with_wind_speed(payload: dict[str, Any], wind_speed_avg_ms: float) -> dict[str, Any]:
    cloned = {**payload, "weather": {**payload["weather"], "wind_speed_avg_ms": wind_speed_avg_ms}}
    return cloned


def _power(value: float | None, exponent: int) -> float:
    return float("nan") if value is None else value ** exponent


def _multiply(left: float | None, right: float | None) -> float:
    return float("nan") if left is None or right is None else left * right


def _ratio(numerator: float | None, denominator: float) -> float:
    if numerator is None or denominator == 0:
        return float("nan")
    return numerator / denominator
