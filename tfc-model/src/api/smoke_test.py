from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.predictor import predict_consumption, predict_wind, unavailable_models


def main() -> None:
    weather = {
        "temp_avg_c": 22.0,
        "temp_max_c": 27.0,
        "temp_min_c": 17.0,
        "humidity_avg_pct": 65.0,
        "pressure_avg_hpa": 1015.0,
        "precip_intensity_avg_mm": 0.0,
        "rain_daily_mm": 0.0,
        "wind_speed_avg_ms": 3.5,
        "wind_speed_max_ms": 6.0,
        "wind_speed_sdev_ms": 1.2,
        "wind_dir_avg_deg": 45.0,
        "wind_dir_max_deg": 90.0,
        "wind_dir_sdev_deg": 20.0,
        "weather_station_count": 3,
    }
    consumption_payload = {
        "municipality_enc": 12,
        "fecha": "2024-06-15",
        "weather": weather,
        "history": {
            "lag_1d": 450.0,
            "lag_7d": 443.0,
            "lag_14d": 448.0,
            "lag_28d": 441.0,
            "rolling_7d_mean": 446.0,
            "rolling_30d_mean": 444.0,
            "rolling_7d_std": 8.0,
        },
    }
    wind_payload = {
        "fecha": "2024-06-15",
        "weather": {
            **weather,
            "wind_speed_avg_ms": 7.5,
            "wind_speed_max_ms": 12.0,
            "wind_speed_sdev_ms": 2.3,
        },
        "canarias_weather_municipality_count": 80,
        "canarias_weather_station_count": 100,
        "history": {
            "lag_1d": 3850.0,
            "lag_2d": 3720.0,
            "lag_3d": 4010.0,
            "rolling_3d_mean": 3860.0,
            "rolling_7d_mean": 3800.0,
            "rolling_7d_std": 420.0,
            "rolling_14d_mean": 3750.0,
            "hidroeolica_lag1": 120.0,
        },
    }

    result = {
        "unavailable_models": unavailable_models(),
        "consumption": predict_consumption(consumption_payload),
        "wind": predict_wind(wind_payload),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
