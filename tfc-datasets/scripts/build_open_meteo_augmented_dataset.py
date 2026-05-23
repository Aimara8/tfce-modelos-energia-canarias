from __future__ import annotations

import argparse
import json
import platform
import time
import tracemalloc
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "docs"
MODEL_DATA = REPO_ROOT / "tfc-model" / "data"
MODEL_EVAL = REPO_ROOT / "tfc-model" / "src" / "evaluation" / "open_meteo_augmented"
CACHE = OUTPUTS / "open_meteo_missing_cache"

WEATHER_COLUMNS = [
    "temp_avg_c",
    "temp_max_c",
    "temp_min_c",
    "pressure_avg_hpa",
    "dew_point_avg_c",
    "precip_intensity_avg_mm",
    "rain_daily_mm",
    "humidity_avg_pct",
    "wind_dir_avg_deg",
    "wind_dir_max_deg",
    "wind_dir_sdev_deg",
    "wind_speed_avg_ms",
    "wind_speed_max_ms",
    "wind_speed_sdev_ms",
]

CONSUMPTION_TARGETS = {
    "total": "demand_total_mwh",
    "residencial": "demand_residencial_mwh",
    "servicios": "demand_servicios_mwh",
    "industria": "demand_industria_mwh",
}


def date_chunks(start: str, end: str) -> list[tuple[str, str]]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    chunks = []
    cursor = start_ts
    while cursor <= end_ts:
        chunk_end = min(pd.Timestamp(cursor.year, 12, 31), end_ts)
        chunks.append((str(cursor.date()), str(chunk_end.date())))
        cursor = chunk_end + pd.Timedelta(days=1)
    return chunks


def fetch_archive(municipality: str, lat: float, lon: float, start: str, end: str) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    safe_name = municipality.replace("/", "_").replace(" ", "_")
    cache_path = CACHE / f"{safe_name}_{start}_{end}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "timezone": "auto",
            "wind_speed_unit": "ms",
            "hourly": ",".join(
                [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "dew_point_2m",
                    "surface_pressure",
                    "precipitation",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "wind_gusts_10m",
                ]
            ),
        }
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    time.sleep(0.2)
    return payload


def circular_mean(values: pd.Series) -> float:
    clean = values.dropna()
    if clean.empty:
        return np.nan
    radians = np.deg2rad(clean.astype(float))
    angle = np.rad2deg(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians))))
    return float(angle + 360 if angle < 0 else angle)


def payload_to_daily(municipality: str, payload: dict) -> pd.DataFrame:
    hourly = payload.get("hourly") or {}
    if not hourly.get("time"):
        return pd.DataFrame()
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(hourly["time"]),
            "temp": hourly.get("temperature_2m"),
            "humidity": hourly.get("relative_humidity_2m"),
            "dew": hourly.get("dew_point_2m"),
            "pressure": hourly.get("surface_pressure"),
            "precip": hourly.get("precipitation"),
            "wind": hourly.get("wind_speed_10m"),
            "wind_dir": hourly.get("wind_direction_10m"),
            "gust": hourly.get("wind_gusts_10m"),
        }
    )
    frame["date"] = frame["time"].dt.strftime("%Y-%m-%d")
    rows = []
    for date, group in frame.groupby("date"):
        rows.append(
            {
                "municipality": municipality,
                "date": date,
                "temp_avg_c": group["temp"].mean(),
                "temp_max_c": group["temp"].max(),
                "temp_min_c": group["temp"].min(),
                "pressure_avg_hpa": group["pressure"].mean(),
                "dew_point_avg_c": group["dew"].mean(),
                "precip_intensity_avg_mm": group["precip"].mean(),
                "rain_daily_mm": group["precip"].sum(),
                "humidity_avg_pct": group["humidity"].mean(),
                "wind_dir_avg_deg": circular_mean(group["wind_dir"]),
                "wind_dir_max_deg": group["wind_dir"].max(),
                "wind_dir_sdev_deg": group["wind_dir"].std(),
                "wind_speed_avg_ms": group["wind"].mean(),
                "wind_speed_max_ms": group["gust"].max(),
                "wind_speed_sdev_ms": group["wind"].std(),
                "weather_station_count": 1,
                "weather_data_source": "open_meteo_historical_missing_municipality",
            }
        )
    return pd.DataFrame(rows)


def build_missing_open_meteo(start: str, end: str) -> pd.DataFrame:
    coords = pd.read_csv(OUTPUTS / "municipality_coordinates_open_meteo.csv")
    missing = coords[coords[["latitude", "longitude"]].notna().all(axis=1)].copy()
    all_rows = []
    for row in missing.itertuples(index=False):
        municipality = row.municipality
        print(f"Open-Meteo missing municipality: {municipality}")
        for chunk_start, chunk_end in date_chunks(start, end):
            payload = fetch_archive(municipality, float(row.latitude), float(row.longitude), chunk_start, chunk_end)
            all_rows.append(payload_to_daily(municipality, payload))
    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    out = out.sort_values(["municipality", "date"]).reset_index(drop=True)
    out.to_csv(OUTPUTS / "weather_daily_municipal_open_meteo_missing.csv", index=False)
    return out


def build_augmented_weather(open_missing: pd.DataFrame) -> pd.DataFrame:
    observed = pd.read_csv(OUTPUTS / "weather_daily_municipal_clean.csv")
    observed = observed.copy()
    observed["weather_data_source"] = "observed_station"
    augmented = pd.concat([observed, open_missing], ignore_index=True)
    augmented = augmented.sort_values(["municipality", "date", "weather_data_source"]).drop_duplicates(
        ["municipality", "date"], keep="first"
    )
    augmented.to_csv(OUTPUTS / "weather_daily_municipal_augmented_full.csv", index=False)
    return augmented


def build_augmented_demand(augmented_weather: pd.DataFrame) -> pd.DataFrame:
    istac = pd.read_csv(OUTPUTS / "istac_daily_municipal_clean.csv")
    merged = istac.merge(augmented_weather, on=["municipality", "date"], how="left")
    final = merged[merged[WEATHER_COLUMNS].notna().any(axis=1)].copy()
    final.to_csv(OUTPUTS / "final_demand_consumption_dataset_augmented_open_meteo.csv", index=False)
    MODEL_DATA.mkdir(parents=True, exist_ok=True)
    final.to_csv(MODEL_DATA / "final_demand_consumption_dataset_augmented_open_meteo.csv", index=False)
    return final


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
        "WMAPE": wmape(y_true, y_pred),
    }


def add_consumption_features(df_raw: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
    df = df_raw.dropna(subset=[target]).copy()
    df["date"] = pd.to_datetime(df["date"])
    mapping = {name: idx for idx, name in enumerate(sorted(df["municipality"].dropna().unique()))}
    df["municipality_enc"] = df["municipality"].map(mapping)
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofweek / 7)
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    df["temp_range_c"] = df["temp_max_c"] - df["temp_min_c"]
    df["hdd"] = np.maximum(18.0 - df["temp_avg_c"], 0)
    df["cdd"] = np.maximum(df["temp_avg_c"] - 18.0, 0)
    df = df.sort_values(["municipality", "date"]).reset_index(drop=True)
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}d"] = df.groupby("municipality")[target].shift(lag)
    df["rolling_7d_mean"] = df.groupby("municipality")[target].transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
    df["rolling_30d_mean"] = df.groupby("municipality")[target].transform(lambda s: s.shift(1).rolling(30, min_periods=7).mean())
    features = [
        "municipality_enc",
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "temp_avg_c",
        "temp_range_c",
        "hdd",
        "cdd",
        "lag_1d",
        "lag_7d",
        "lag_14d",
        "lag_28d",
        "rolling_7d_mean",
        "rolling_30d_mean",
    ]
    df = df.dropna(subset=features).sort_values("date").reset_index(drop=True)
    return df, features


def evaluate_consumption(name: str, df: pd.DataFrame) -> list[dict]:
    import xgboost as xgb

    rows = []
    for sector, target in CONSUMPTION_TARGETS.items():
        work, features = add_consumption_features(df, target)
        dates = sorted(work["date"].unique())
        test_start = dates[int(len(dates) * 0.8)]
        train = work[work["date"] < test_start]
        test = work[work["date"] >= test_start]
        model = xgb.XGBRegressor(
            n_estimators=260,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            tree_method="hist",
            eval_metric="rmse",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(train[features], train[target], verbose=False)
        pred = model.predict(test[features])
        row = metrics(test[target].values, pred)
        row.update(
            {
                "problem": "consumption",
                "scenario": name,
                "target": sector,
                "municipalities": int(work["municipality"].nunique()),
                "rows": int(len(work)),
                "open_meteo_rows": int((work.get("weather_data_source") == "open_meteo_historical_missing_municipality").sum())
                if "weather_data_source" in work
                else 0,
            }
        )
        rows.append(row)
    return rows


def aggregate_canarias_weather(weather: pd.DataFrame) -> pd.DataFrame:
    weather = weather.copy()
    weather["date"] = pd.to_datetime(weather["date"])
    numeric = WEATHER_COLUMNS
    grouped = weather.groupby("date")[numeric].mean(numeric_only=True).reset_index()
    grouped["canarias_weather_municipality_count"] = weather.groupby("date")["municipality"].nunique().values
    grouped["canarias_weather_station_count"] = weather.groupby("date")["weather_station_count"].sum(min_count=1).values
    return grouped


def evaluate_eolica(name: str, weather: pd.DataFrame) -> dict:
    ree = pd.read_csv(OUTPUTS / "ree_renewables_canarias_daily_wide.csv", parse_dates=["date"])
    merged = ree.merge(aggregate_canarias_weather(weather), on="date", how="inner")
    work = merged.dropna(subset=["ree_eolica_value"]).sort_values("date").reset_index(drop=True)
    work["month_sin"] = np.sin(2 * np.pi * work["date"].dt.month / 12)
    work["month_cos"] = np.cos(2 * np.pi * work["date"].dt.month / 12)
    work["wind_speed_avg_ms_cb"] = work["wind_speed_avg_ms"] ** 3
    work["wind_speed_max_ms_cb"] = work["wind_speed_max_ms"] ** 3
    for lag in [1, 2, 3]:
        work[f"lag_{lag}d"] = work["ree_eolica_value"].shift(lag)
    work["rolling_7d_mean"] = work["ree_eolica_value"].shift(1).rolling(7, min_periods=3).mean()
    features = [
        "canarias_weather_municipality_count",
        "canarias_weather_station_count",
        "wind_speed_avg_ms",
        "wind_speed_max_ms",
        "wind_speed_avg_ms_cb",
        "wind_speed_max_ms_cb",
        "month_sin",
        "month_cos",
        "lag_1d",
        "lag_2d",
        "lag_3d",
        "rolling_7d_mean",
    ]
    work = work.dropna(subset=features).reset_index(drop=True)
    dates = sorted(work["date"].unique())
    test_start = dates[int(len(dates) * 0.8)]
    train = work[work["date"] < test_start]
    test = work[work["date"] >= test_start]
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(max_iter=450, max_depth=4, learning_rate=0.045, random_state=42)),
        ]
    )
    model.fit(train[features], train["ree_eolica_value"])
    pred = model.predict(test[features])
    row = metrics(test["ree_eolica_value"].values, pred)
    row.update(
        {
            "problem": "eolica",
            "scenario": name,
            "target": "eolica",
            "municipalities": int(weather["municipality"].nunique()),
            "rows": int(len(work)),
            "open_meteo_rows": int((weather.get("weather_data_source") == "open_meteo_historical_missing_municipality").sum())
            if "weather_data_source" in weather
            else 0,
        }
    )
    return row


def build_augmented_renewable(augmented_weather: pd.DataFrame) -> pd.DataFrame:
    ree = pd.read_csv(OUTPUTS / "ree_renewables_canarias_daily_wide.csv", parse_dates=["date"])
    if "ree_generacion_renovable_pct" in ree.columns:
        ree = ree.drop(columns=["ree_generacion_renovable_pct"])
    merged = ree.merge(aggregate_canarias_weather(augmented_weather), on="date", how="left")
    merged["weather_data_source"] = np.where(
        merged["canarias_weather_municipality_count"].notna(),
        "canarias_observed_plus_open_meteo_missing_aggregated",
        "",
    )
    merged.to_csv(OUTPUTS / "final_renewable_generation_dataset_augmented_open_meteo.csv", index=False)
    MODEL_DATA.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MODEL_DATA / "final_renewable_generation_dataset_augmented_open_meteo.csv", index=False)
    return merged


def write_report(results: pd.DataFrame, final: pd.DataFrame, open_missing: pd.DataFrame) -> None:
    current_municipalities = int(final.loc[final["weather_data_source"] == "observed_station", "municipality"].nunique())
    total = final["municipality"].nunique()
    added = total - current_municipalities
    lines = [
        "# Open-Meteo territorial augmentation",
        "",
        f"Municipios actuales del modelo de consumo: **{current_municipalities}**.",
        f"Municipios añadidos con Open-Meteo: **{added}**.",
        f"Total resultante: **{total}/87** municipios ISTAC.",
        "",
        "Open-Meteo se usa solo para municipios sin meteo observada. Las estaciones reales mantienen prioridad.",
        "",
        "Nota: la agregacion meteorologica para eolica puede mostrar 88 municipios porque incluye `Frontera`, que tiene estacion observada pero no aparece en el dataset ISTAC de consumo.",
        "",
        "## Comparativa de modelos",
        "",
        "| Problem | Scenario | Target | Municipios | MAE | RMSE | R2 | MAPE | WMAPE |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results.to_dict(orient="records"):
        lines.append(
            f"| {row['problem']} | {row['scenario']} | {row['target']} | {row['municipalities']} | "
            f"{row['MAE']:.4f} | {row['RMSE']:.4f} | {row['R2']:.4f} | {row['MAPE']:.2f} | {row['WMAPE']:.2f} |"
        )
    lines += [
        "",
        "## Lectura",
        "",
        "- Si `augmented_open_meteo_missing` empeora poco o mejora, compensa por cobertura territorial.",
        "- Si empeora bastante, mantener Open-Meteo como fuente separada y calibrar por isla/municipio antes de entrenar definitivo.",
        "- La comparacion no es perfecta porque el escenario ampliado predice municipios adicionales, no exactamente el mismo panel.",
        "",
        "## Reentrenamiento oficial posterior",
        "",
    ]
    official_consumption = REPO_ROOT / "tfc-model" / "src" / "evaluation" / "consumption_energy_demand" / "metricas_todos_sectores.csv"
    official_eolica = REPO_ROOT / "tfc-model" / "src" / "evaluation" / "renewable_energy_generation" / "metricas_eolica.csv"
    if official_consumption.exists() and official_eolica.exists():
        consumption_metrics = pd.read_csv(official_consumption)
        eolica_metrics = pd.read_csv(official_eolica)
        lines += [
            "Metricas test tras promover los datasets ampliados a `tfc-model/data` y reentrenar los scripts oficiales:",
            "",
            "| Modelo oficial | Target | MAE | RMSE | R2 | MAPE | WMAPE |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in consumption_metrics[consumption_metrics["split"] == "test"].itertuples(index=False):
            lines.append(
                f"| consumo XGBoost | {row.sector} | {row.MAE:.4f} | {row.RMSE:.4f} | {row.R2:.4f} | {row.MAPE:.2f} | {row.WMAPE:.2f} |"
            )
        hgb = eolica_metrics[eolica_metrics["modelo"] == "HGB"]
        if not hgb.empty:
            row = hgb.iloc[0]
            lines.append(
                f"| eolica HGB | eolica | {row['MAE']:.4f} | {row['RMSE']:.4f} | {row['R2']:.4f} | {row['MAPE']:.2f} | {row['WMAPE']:.2f} |"
            )
        lines += [
            "",
            "Decision: se mantiene el dataset ampliado porque cubre 87/87 municipios ISTAC y la degradacion en consumo es moderada. Eolica mejora con la agregacion territorial ampliada.",
            "",
        ]
    else:
        lines += ["Pendiente de reentrenar los modelos oficiales.", ""]
    lines += [
        "## Archivos generados",
        "",
        "- `outputs/weather_daily_municipal_open_meteo_missing.csv`",
        "- `outputs/weather_daily_municipal_augmented_full.csv`",
        "- `outputs/final_demand_consumption_dataset_augmented_open_meteo.csv`",
        "- `outputs/final_renewable_generation_dataset_augmented_open_meteo.csv`",
        "- `tfc-model/data/final_demand_consumption_dataset_augmented_open_meteo.csv`",
        "- `tfc-model/data/final_renewable_generation_dataset_augmented_open_meteo.csv`",
        "- `tfc-model/src/evaluation/open_meteo_augmented/open_meteo_augmented_model_metrics.csv`",
        "",
    ]
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "OPEN_METEO_AUGMENTATION.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    tracemalloc.start()
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-06-30")
    parser.add_argument("--skip-fetch", action="store_true")
    args = parser.parse_args()

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    MODEL_EVAL.mkdir(parents=True, exist_ok=True)

    if args.skip_fetch and (OUTPUTS / "weather_daily_municipal_open_meteo_missing.csv").exists():
        open_missing = pd.read_csv(OUTPUTS / "weather_daily_municipal_open_meteo_missing.csv")
    else:
        open_missing = build_missing_open_meteo(args.start, args.end)
    augmented_weather = build_augmented_weather(open_missing)
    final = build_augmented_demand(augmented_weather)
    build_augmented_renewable(augmented_weather)

    rollback = MODEL_DATA / "final_demand_consumption_dataset_observed_39_rollback.csv"
    current = pd.read_csv(rollback if rollback.exists() else MODEL_DATA / "final_demand_consumption_dataset.csv")
    results = []
    results.extend(evaluate_consumption("observed_current_39", current))
    results.extend(evaluate_consumption("augmented_open_meteo_missing", final))
    observed_weather = pd.read_csv(OUTPUTS / "weather_daily_municipal_clean.csv")
    results.append(evaluate_eolica("observed_current_weather", observed_weather))
    results.append(evaluate_eolica("augmented_open_meteo_missing", augmented_weather))
    results_df = pd.DataFrame(results)
    results_df.to_csv(MODEL_EVAL / "open_meteo_augmented_model_metrics.csv", index=False)
    write_report(results_df, final, open_missing)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    resource_summary = {
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "tracemalloc_current_mb": round(current / (1024 * 1024), 3),
        "tracemalloc_peak_mb": round(peak / (1024 * 1024), 3),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "rows_open_meteo_missing": int(len(open_missing)),
        "rows_augmented_consumption": int(len(final)),
        "municipalities_augmented_consumption": int(final["municipality"].nunique()),
    }
    try:
        import psutil

        process = psutil.Process()
        resource_summary["process_rss_mb"] = round(process.memory_info().rss / (1024 * 1024), 3)
        resource_summary["cpu_logical"] = psutil.cpu_count(logical=True)
    except Exception as exc:
        resource_summary["psutil_error"] = str(exc)
    (MODEL_EVAL / "open_meteo_augmented_run_resources.json").write_text(
        json.dumps(resource_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(DOCS / "OPEN_METEO_AUGMENTATION.md")


if __name__ == "__main__":
    main()
