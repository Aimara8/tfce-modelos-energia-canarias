from __future__ import annotations

from datetime import date
from functools import lru_cache
from statistics import mean
from typing import Any

import numpy as np
import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

MUNICIPALITY_COORDS = {
    "Adeje": (28.1227, -16.7260), "Agaete": (28.1000, -15.7000), "Alajeró": (28.0621, -17.2407),
    "Arona": (28.0996, -16.6810), "Arrecife": (28.9630, -13.5477), "Artenara": (28.0206, -15.6469),
    "Arucas": (28.1198, -15.5231), "Betancuria": (28.4240, -14.0560), "Breña Baja": (28.6300, -17.7900),
    "El Pinar de El Hierro": (27.7250, -17.9850), "Fuencaliente de La Palma": (28.4880, -17.8460),
    "Garachico": (28.3733, -16.7634), "Haría": (29.1454, -13.4994), "Hermigua": (28.1674, -17.1909),
    "La Guancha": (28.3732, -16.6510), "La Orotava": (28.3908, -16.5231),
    "Las Palmas de Gran Canaria": (28.1235, -15.4363), "Mogán": (27.8839, -15.7254),
    "Moya": (28.1110, -15.5820), "Puerto del Rosario": (28.5004, -13.8627),
    "Puntagorda": (28.7740, -17.9780), "Pájara": (28.3500, -14.1070),
    "San Bartolomé de Tirajana": (27.9248, -15.5733), "San Cristóbal de La Laguna": (28.4874, -16.3159),
    "San Sebastián de La Gomera": (28.0916, -17.1133), "Santa Cruz de La Palma": (28.6835, -17.7642),
    "Santa Cruz de Tenerife": (28.4636, -16.2518), "Santa Lucía de Tirajana": (27.9117, -15.5407),
    "Santa María de Guía de Gran Canaria": (28.1397, -15.6329), "Santiago del Teide": (28.2944, -16.8168),
    "Teguise": (29.0605, -13.5598), "Telde": (27.9955, -15.4174), "Tuineje": (28.3231, -14.0477),
    "Vallehermoso": (28.1796, -17.2638), "Valverde": (27.8099, -17.9158),
    "Vega de San Mateo": (28.0089, -15.5329), "Vilaflor de Chasna": (28.1562, -16.6359),
    "Villa de Mazo": (28.6090, -17.7780), "Yaiza": (28.9529, -13.7656),
}

CANARY_REPRESENTATIVE_POINTS = [
    MUNICIPALITY_COORDS["Las Palmas de Gran Canaria"],
    MUNICIPALITY_COORDS["Santa Cruz de Tenerife"],
    MUNICIPALITY_COORDS["Arrecife"],
    MUNICIPALITY_COORDS["Puerto del Rosario"],
    MUNICIPALITY_COORDS["Santa Cruz de La Palma"],
    MUNICIPALITY_COORDS["San Sebastián de La Gomera"],
    MUNICIPALITY_COORDS["Valverde"],
    MUNICIPALITY_COORDS["San Bartolomé de Tirajana"],
]


def forecast_for_municipality(municipality: str, target_date: str | date) -> dict[str, Any]:
    if municipality not in MUNICIPALITY_COORDS:
        raise ValueError(f"No hay coordenadas configuradas para {municipality}.")
    lat, lon = MUNICIPALITY_COORDS[municipality]
    return _forecast_for_point(lat, lon, str(target_date), station_count=1)


def forecast_for_canarias(target_date: str | date) -> dict[str, Any]:
    forecasts = [
        _forecast_for_point(lat, lon, str(target_date), station_count=1)
        for lat, lon in CANARY_REPRESENTATIVE_POINTS
    ]
    numeric_keys = [key for key, value in forecasts[0].items() if isinstance(value, (int, float))]
    aggregated = {}
    for key in numeric_keys:
        values = [item[key] for item in forecasts if item.get(key) is not None]
        aggregated[key] = float(mean(values)) if values else None
    aggregated["weather_station_count"] = len(forecasts)
    return aggregated


@lru_cache(maxsize=512)
def _forecast_for_point(lat: float, lon: float, target_date: str, station_count: int) -> dict[str, Any]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": target_date,
        "end_date": target_date,
        "timezone": "auto",
        "wind_speed_unit": "ms",
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "pressure_msl",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
        ]),
    }
    response = requests.get(OPEN_METEO_URL, params=params, timeout=12)
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly") or {}
    if not hourly.get("time"):
        raise ValueError(f"Open-Meteo no devolvio prediccion horaria para {target_date}.")

    temp = _clean(hourly.get("temperature_2m", []))
    humidity = _clean(hourly.get("relative_humidity_2m", []))
    dew = _clean(hourly.get("dew_point_2m", []))
    pressure = _clean(hourly.get("pressure_msl", []))
    precip = _clean(hourly.get("precipitation", []))
    wind = _clean(hourly.get("wind_speed_10m", []))
    wind_dir = _clean(hourly.get("wind_direction_10m", []))
    gusts = _clean(hourly.get("wind_gusts_10m", []))

    return {
        "temp_avg_c": float(np.mean(temp)),
        "temp_max_c": float(np.max(temp)),
        "temp_min_c": float(np.min(temp)),
        "humidity_avg_pct": float(np.mean(humidity)),
        "dew_point_avg_c": float(np.mean(dew)),
        "pressure_avg_hpa": float(np.mean(pressure)),
        "precip_intensity_avg_mm": float(np.mean(precip)) if precip else 0.0,
        "rain_daily_mm": float(np.sum(precip)) if precip else 0.0,
        "wind_speed_avg_ms": float(np.mean(wind)),
        "wind_speed_max_ms": float(np.max(gusts or wind)),
        "wind_speed_sdev_ms": float(np.std(wind)),
        "wind_dir_avg_deg": _circular_mean(wind_dir),
        "wind_dir_max_deg": float(np.max(wind_dir)) if wind_dir else None,
        "wind_dir_sdev_deg": float(np.std(wind_dir)) if wind_dir else None,
        "weather_station_count": station_count,
    }


def _clean(values: list[Any]) -> list[float]:
    return [float(value) for value in values if value is not None]


def _circular_mean(values: list[float]) -> float | None:
    if not values:
        return None
    radians = np.deg2rad(values)
    angle = np.rad2deg(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians))))
    return float(angle + 360 if angle < 0 else angle)
