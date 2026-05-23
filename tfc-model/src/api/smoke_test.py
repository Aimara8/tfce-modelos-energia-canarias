from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.predictor import predict_consumption, predict_wind, project_metadata, unavailable_models


def main() -> None:
    metadata = project_metadata()
    municipality = metadata["consumption"]["municipalities"][0]
    fecha = metadata["consumption"]["date_max"]
    weather = {
        "temp_avg_c": 23.0,
        "temp_max_c": 27.0,
        "temp_min_c": 19.0,
        "humidity_avg_pct": 66.0,
        "pressure_avg_hpa": 1015.0,
        "precip_intensity_avg_mm": 0.0,
        "rain_daily_mm": 0.0,
        "wind_speed_avg_ms": 5.0,
        "wind_speed_max_ms": 9.0,
        "wind_speed_sdev_ms": 1.5,
        "wind_dir_avg_deg": 180.0,
        "wind_dir_max_deg": 220.0,
        "wind_dir_sdev_deg": 35.0,
        "weather_station_count": 3,
    }
    result = {
        "unavailable_models": unavailable_models(),
        "metadata_summary": {
            "municipality": municipality,
            "fecha": fecha,
            "wind_range": [metadata["renewable"]["date_min"], metadata["renewable"]["date_max"]],
        },
        "consumption": predict_consumption({"municipality": municipality, "fecha": fecha, "weather": weather}),
        "wind": predict_wind({
            "fecha": metadata["renewable"]["date_max"],
            "weather": {**weather, "wind_speed_avg_ms": 7.5, "wind_speed_max_ms": 12.0, "wind_speed_sdev_ms": 2.2},
            "canarias_weather_municipality_count": 87,
            "canarias_weather_station_count": 50,
        }),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
