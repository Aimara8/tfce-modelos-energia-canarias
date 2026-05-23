from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import os
import platform
import time
import tracemalloc

import joblib
import numpy as np
import pandas as pd

from .external_weather import forecast_for_canarias, forecast_for_municipality
from .feature_engineering import (
    CONSUMPTION_FEATURES,
    CONSUMPTION_TARGETS,
    WIND_FEATURES,
    build_consumption_features,
    build_wind_features,
    with_wind_speed,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "src" / "models"
EVAL_DIR = ROOT_DIR / "src" / "evaluation"

CONSUMPTION_MODEL_FILES = {
    "total": MODELS_DIR / "consumption_energy_demand" / "xgboost_total.json",
    "residencial": MODELS_DIR / "consumption_energy_demand" / "xgboost_residencial.json",
    "servicios": MODELS_DIR / "consumption_energy_demand" / "xgboost_servicios.json",
    "industria": MODELS_DIR / "consumption_energy_demand" / "xgboost_industria.json",
}
WIND_MODEL_FILE = MODELS_DIR / "renewable_energy_generation" / "hgb_eolica.pkl"
WIND_IMPUTER_FILE = MODELS_DIR / "renewable_energy_generation" / "imputer_eolica.pkl"

_models: dict[str, Any] = {}
_model_errors: dict[str, str] = {}


def _load_consumption_models() -> None:
    try:
        import xgboost as xgb
    except Exception as exc:  # pragma: no cover
        for sector in CONSUMPTION_MODEL_FILES:
            _model_errors[f"consumo_{sector}"] = f"xgboost no disponible: {exc}"
        return

    for sector, path in CONSUMPTION_MODEL_FILES.items():
        key = f"consumo_{sector}"
        if key in _models or key in _model_errors:
            continue
        if not path.exists():
            _model_errors[key] = f"no encontrado: {path}"
            continue
        model = xgb.XGBRegressor()
        model.load_model(str(path))
        expected = getattr(model, "n_features_in_", len(CONSUMPTION_FEATURES))
        if expected != len(CONSUMPTION_FEATURES):
            _model_errors[key] = f"features esperadas={expected}, construidas={len(CONSUMPTION_FEATURES)}"
            continue
        _models[key] = model


def _load_wind_model() -> None:
    if "eolica" in _models or "eolica" in _model_errors:
        return
    if not WIND_MODEL_FILE.exists():
        _model_errors["eolica"] = f"no encontrado: {WIND_MODEL_FILE}"
        return
    if not WIND_IMPUTER_FILE.exists():
        _model_errors["eolica"] = f"imputador no encontrado: {WIND_IMPUTER_FILE}"
        return
    try:
        model = joblib.load(WIND_MODEL_FILE)
        imputer = joblib.load(WIND_IMPUTER_FILE)
    except Exception as exc:
        _model_errors["eolica"] = f"error al cargar pkl: {exc}"
        return
    expected = getattr(model, "n_features_in_", len(WIND_FEATURES))
    if expected != len(WIND_FEATURES):
        _model_errors["eolica"] = f"features esperadas={expected}, construidas={len(WIND_FEATURES)}"
        return
    _models["eolica"] = {"model": model, "imputer": imputer}


def ensure_models_loaded() -> None:
    _load_consumption_models()
    _load_wind_model()


def loaded_model_names() -> list[str]:
    ensure_models_loaded()
    return sorted(_models)


def unavailable_models() -> dict[str, str]:
    ensure_models_loaded()
    return dict(sorted(_model_errors.items()))


@lru_cache(maxsize=1)
def demand_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "final_demand_consumption_dataset.csv", parse_dates=["date"])
    return df.sort_values(["municipality", "date"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def renewable_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "final_renewable_generation_dataset.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


@lru_cache(maxsize=1)
def municipality_mapping() -> dict[str, int]:
    names = sorted(demand_dataset()["municipality"].dropna().unique().tolist())
    return {name: index for index, name in enumerate(names)}


def project_metadata() -> dict[str, Any]:
    demand = demand_dataset()
    renewable = renewable_dataset()
    consumption_metrics = _read_csv_records(EVAL_DIR / "consumption_energy_demand" / "metricas_todos_sectores.csv")
    wind_metrics = _read_csv_records(EVAL_DIR / "renewable_energy_generation" / "metricas_eolica.csv")
    return {
        "consumption": {
            "municipalities": list(municipality_mapping().keys()),
            "feature_count": len(CONSUMPTION_FEATURES),
            "date_min": str(demand["date"].min().date()),
            "date_max": str(demand["date"].max().date()),
            "sectors": list(CONSUMPTION_TARGETS.keys()),
        },
        "renewable": {
            "feature_count": len(WIND_FEATURES),
            "date_min": str(renewable["date"].min().date()),
            "date_max": str(renewable["date"].max().date()),
            "target": "ree_eolica_value",
            "known_limit": "Meteorologia agregada diaria de Canarias; no incluye potencia instalada ni curtailment.",
            "forecast_source": "Open-Meteo forecast API when date is not present in local historical dataset.",
        },
        "evaluation": {
            "consumption_metrics": consumption_metrics,
            "wind_metrics": wind_metrics,
            "wind_monthly": wind_monthly_diagnostics(),
            "wind_worst_days": wind_worst_days(),
            "model_status": {"loaded": loaded_model_names(), "unavailable": unavailable_models()},
        },
    }


def consumption_dashboard() -> dict[str, Any]:
    df = demand_dataset().copy()
    metrics = _read_csv_records(EVAL_DIR / "consumption_energy_demand" / "metricas_todos_sectores.csv")
    sector_cols = CONSUMPTION_TARGETS
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date]
    sector_totals = [
        {"sector": sector, "mwh": round(float(latest[col].sum()), 2)}
        for sector, col in sector_cols.items()
    ]
    monthly = df.assign(month=df["date"].dt.strftime("%Y-%m")).groupby("month")["demand_total_mwh"].sum().tail(18)
    top_municipalities = (
        latest.groupby("municipality")["demand_total_mwh"].sum().sort_values(ascending=False).head(10)
    )
    return {
        "kpis": {
            "municipalities": int(df["municipality"].nunique()),
            "rows": int(len(df)),
            "date_min": str(df["date"].min().date()),
            "date_max": str(latest_date.date()),
            "latest_total_mwh": round(float(latest["demand_total_mwh"].sum()), 2),
        },
        "sector_totals": sector_totals,
        "monthly_total": [{"month": key, "mwh": round(float(value), 2)} for key, value in monthly.items()],
        "top_municipalities": [
            {"municipality": key, "mwh": round(float(value), 2)}
            for key, value in top_municipalities.items()
        ],
        "metrics": metrics,
    }


def wind_dashboard() -> dict[str, Any]:
    df = renewable_dataset().copy()
    latest = df.iloc[-1]
    tech_cols = {
        "eolica": "ree_eolica_value",
        "solar": "ree_solar_fotovoltaica_value",
        "hidroeolica": "ree_hidroeolica_value",
        "hidraulica": "ree_hidraulica_value",
        "otras": "ree_otras_renovables_value",
    }
    mix = [
        {"technology": name, "mwh": round(float(latest[col]), 2)}
        for name, col in tech_cols.items()
        if col in df.columns and pd.notna(latest[col])
    ]
    monthly = df.assign(month=df["date"].dt.strftime("%Y-%m")).groupby("month")[
        ["ree_eolica_value", "ree_solar_fotovoltaica_value", "wind_speed_avg_ms"]
    ].mean().tail(18)
    metrics = _read_csv_records(EVAL_DIR / "renewable_energy_generation" / "metricas_eolica.csv")
    baseline = _read_csv_records(EVAL_DIR / "renewable_energy_generation" / "baseline_eolica.csv")
    return {
        "kpis": {
            "rows": int(len(df)),
            "date_min": str(df["date"].min().date()),
            "date_max": str(df["date"].max().date()),
            "latest_eolica_mwh": round(float(latest["ree_eolica_value"]), 2),
            "latest_wind_ms": round(float(latest["wind_speed_avg_ms"]), 2),
            "weather_stations": int(latest["canarias_weather_station_count"]),
        },
        "latest_mix": mix,
        "monthly": [
            {
                "month": idx,
                "eolica_mwh": round(float(row["ree_eolica_value"]), 2),
                "solar_mwh": round(float(row["ree_solar_fotovoltaica_value"]), 2),
                "wind_ms": round(float(row["wind_speed_avg_ms"]), 2),
            }
            for idx, row in monthly.iterrows()
        ],
        "metrics": metrics,
        "baseline": baseline,
        "monthly_errors": wind_monthly_diagnostics(),
        "worst_days": wind_worst_days(),
    }


def benchmark_api() -> dict[str, Any]:
    metadata = project_metadata()
    municipality = metadata["consumption"]["municipalities"][0]
    weather = _consumption_weather(municipality, metadata["consumption"]["date_max"], None)
    wind_weather = _wind_weather(metadata["renewable"]["date_max"], None)
    samples = []
    tracemalloc.start()
    started = time.perf_counter()
    for _ in range(5):
        t0 = time.perf_counter()
        predict_consumption({"municipality": municipality, "fecha": metadata["consumption"]["date_max"], "weather": weather})
        samples.append({"operation": "predict_consumption", "latency_ms": (time.perf_counter() - t0) * 1000})
        t0 = time.perf_counter()
        predict_wind({
            "fecha": metadata["renewable"]["date_max"],
            "weather": wind_weather,
            "canarias_weather_municipality_count": 87,
            "canarias_weather_station_count": 50,
        })
        samples.append({"operation": "predict_eolica", "latency_ms": (time.perf_counter() - t0) * 1000})
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    process_mb = None
    cpu_percent = None
    try:
        import psutil

        process = psutil.Process(os.getpid())
        process_mb = round(process.memory_info().rss / (1024 * 1024), 2)
        cpu_percent = psutil.cpu_percent(interval=0.1)
    except Exception:
        pass
    summary = {}
    for operation in sorted({item["operation"] for item in samples}):
        latencies = [item["latency_ms"] for item in samples if item["operation"] == operation]
        summary[operation] = {
            "avg_ms": round(float(sum(latencies) / len(latencies)), 2),
            "min_ms": round(float(min(latencies)), 2),
            "max_ms": round(float(max(latencies)), 2),
        }
    return {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "process_rss_mb": process_mb,
            "cpu_percent_sample": cpu_percent,
        },
        "summary": summary,
        "samples": [{**item, "latency_ms": round(item["latency_ms"], 2)} for item in samples],
        "elapsed_total_ms": round((time.perf_counter() - started) * 1000, 2),
        "tracemalloc_peak_mb": round(peak / (1024 * 1024), 2),
    }


def predict_consumption(payload: dict[str, Any]) -> dict[str, Any]:
    ensure_models_loaded()
    municipality = payload["municipality"]
    mapping = municipality_mapping()
    if municipality not in mapping:
        raise ValueError(f"Municipio no reconocido: {municipality}")

    warnings = []
    predictions = []
    references = []
    weather = _consumption_weather(municipality, payload["fecha"], payload.get("weather"))
    for sector, target in CONSUMPTION_TARGETS.items():
        key = f"consumo_{sector}"
        if key not in _models:
            predictions.append({"sector": sector, "mwh": None, "baseline_mwh": None, "history_source": "manual"})
            continue
        history, source = _consumption_history(municipality, payload["fecha"], target, payload.get("history"))
        if source == "manual":
            warnings.append(f"{sector}: se usa historico manual porque no hay suficientes datos previos.")
        features = build_consumption_features(
            fecha=payload["fecha"],
            municipality_enc=mapping[municipality],
            weather=weather,
            history=history,
        )
        pred = float(_models[key].predict(features.to_numpy(dtype=float))[0])
        baseline = float(history["rolling_7d_mean"])
        predictions.append({
            "sector": sector,
            "mwh": round(max(pred, 0.0), 4),
            "baseline_mwh": round(max(baseline, 0.0), 4),
            "history_source": source,
        })
        if sector == "total":
            references = [
                {"label": "ayer", "value": history["lag_1d"]},
                {"label": "hace 7 dias", "value": history["lag_7d"]},
                {"label": "media 7d", "value": history["rolling_7d_mean"]},
                {"label": "media 30d", "value": history["rolling_30d_mean"]},
            ]

    return {
        "municipality_enc": mapping[municipality],
        "predictions": predictions,
        "chart_bars": [{"label": item["sector"], "value": item["mwh"]} for item in predictions],
        "chart_reference": references,
        "warnings": sorted(set(warnings)),
        "model_status": _status_for(["consumo_total", "consumo_residencial", "consumo_servicios", "consumo_industria"]),
    }


def predict_wind(payload: dict[str, Any]) -> dict[str, Any]:
    ensure_models_loaded()
    history, source = _wind_history(payload["fecha"], payload.get("history"))
    weather = _wind_weather(payload["fecha"], payload.get("weather"))
    payload = {**payload, "history": history, "weather": weather}
    warnings = []
    if source == "manual":
        warnings.append("Se usa historico eolico manual porque no hay suficientes datos previos.")

    chart_series = [
        {"label": "hace 3 dias", "value": history["lag_3d"]},
        {"label": "hace 2 dias", "value": history["lag_2d"]},
        {"label": "ayer", "value": history["lag_1d"]},
    ]
    if "eolica" not in _models:
        return {
            "eolica_predicha_mwh": None,
            "uncertainty_low_mwh": None,
            "uncertainty_high_mwh": None,
            "confidence": "sin_modelo",
            "condition": "sin_modelo",
            "comparison_to_rolling_7d_pct": None,
            "chart_series": chart_series + [{"label": "prediccion", "value": None}],
            "sensitivity_by_wind": [],
            "warnings": warnings,
            "model_status": _status_for(["eolica"]),
        }

    pred = _predict_wind_raw(payload)
    uncertainty = _wind_uncertainty(pred, payload["fecha"])
    chart_series.append({"label": "prediccion", "value": pred})
    rolling = history["rolling_7d_mean"]
    comparison = None if rolling == 0 else round((pred - rolling) / rolling * 100, 2)
    if uncertainty["confidence"] == "baja":
        warnings.append("Confianza baja: la eolica tiene mayor error historico y depende de factores no observados.")
    return {
        "eolica_predicha_mwh": pred,
        "uncertainty_low_mwh": uncertainty["low"],
        "uncertainty_high_mwh": uncertainty["high"],
        "confidence": uncertainty["confidence"],
        "condition": _wind_condition(payload["weather"].get("wind_speed_avg_ms"), pred, rolling),
        "comparison_to_rolling_7d_pct": comparison,
        "chart_series": chart_series,
        "sensitivity_by_wind": _wind_sensitivity(payload),
        "warnings": warnings,
        "model_status": _status_for(["eolica"]),
    }


def wind_monthly_diagnostics() -> list[dict[str, Any]]:
    path = EVAL_DIR / "renewable_energy_generation" / "predicciones_test_eolica.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path, parse_dates=["date"])
    if "pred_hgb" not in df.columns:
        return []
    df["abs_err"] = (df["ree_eolica_value"] - df["pred_hgb"]).abs()
    df["bias"] = df["pred_hgb"] - df["ree_eolica_value"]
    df["month"] = df["date"].dt.strftime("%Y-%m")
    rows = []
    for month, group in df.groupby("month"):
        denom = group["ree_eolica_value"].abs().sum()
        rows.append({
            "month": month,
            "wmape": None if denom == 0 else round(group["abs_err"].sum() / denom * 100, 2),
            "bias_mwh": round(group["bias"].mean(), 2),
            "real_mean_mwh": round(group["ree_eolica_value"].mean(), 2),
            "pred_mean_mwh": round(group["pred_hgb"].mean(), 2),
        })
    return rows


def wind_worst_days(limit: int = 10) -> list[dict[str, Any]]:
    path = EVAL_DIR / "renewable_energy_generation" / "predicciones_test_eolica.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path, parse_dates=["date"])
    if "pred_hgb" not in df.columns:
        return []
    df["abs_err"] = (df["ree_eolica_value"] - df["pred_hgb"]).abs()
    out = df.sort_values("abs_err", ascending=False).head(limit)
    return [
        {
            "date": str(row.date.date()),
            "real_mwh": round(float(row.ree_eolica_value), 2),
            "pred_mwh": round(float(row.pred_hgb), 2),
            "abs_err_mwh": round(float(row.abs_err), 2),
        }
        for row in out.itertuples()
    ]


def _consumption_history(
    municipality: str,
    fecha: str,
    target: str,
    manual_history: dict[str, Any] | None,
) -> tuple[dict[str, float], str]:
    df = demand_dataset()
    date_value = pd.Timestamp(fecha)
    hist = df[(df["municipality"] == municipality) & (df["date"] < date_value)].sort_values("date")
    if len(hist) >= 30 and target in hist.columns:
        series = hist[target].dropna()
        if len(series) >= 30:
            return {
                "lag_1d": float(series.iloc[-1]),
                "lag_7d": float(series.iloc[-7]),
                "lag_14d": float(series.iloc[-14]),
                "lag_28d": float(series.iloc[-28]),
                "rolling_7d_mean": float(series.iloc[-7:].mean()),
                "rolling_30d_mean": float(series.iloc[-30:].mean()),
                "rolling_7d_std": float(series.iloc[-7:].std()),
            }, "historical"
    if manual_history:
        return {key: float(value) for key, value in manual_history.items()}, "manual"
    raise ValueError(f"No hay suficiente historico para {municipality} antes de {fecha}.")


def _consumption_weather(
    municipality: str,
    fecha: str,
    manual_weather: dict[str, Any] | None,
) -> dict[str, Any]:
    if manual_weather:
        return manual_weather
    df = demand_dataset()
    date_value = pd.Timestamp(fecha)
    row = df[(df["municipality"] == municipality) & (df["date"] == date_value)]
    if row.empty:
        try:
            return forecast_for_municipality(municipality, fecha)
        except Exception as exc:
            raise ValueError(
                f"No hay meteorologia historica para {municipality} en {fecha} y no se pudo consultar Open-Meteo: {exc}. "
                "Introduce un escenario meteorologico manual."
            ) from exc
    return _weather_from_row(row.iloc[0], municipal=True)


def _wind_history(fecha: str, manual_history: dict[str, Any] | None) -> tuple[dict[str, float], str]:
    df = renewable_dataset()
    date_value = pd.Timestamp(fecha)
    hist = df[df["date"] < date_value].sort_values("date")
    if len(hist) >= 14:
        series = hist["ree_eolica_value"].dropna()
        if len(series) >= 14:
            return {
                "lag_1d": float(series.iloc[-1]),
                "lag_2d": float(series.iloc[-2]),
                "lag_3d": float(series.iloc[-3]),
                "rolling_3d_mean": float(series.iloc[-3:].mean()),
                "rolling_7d_mean": float(series.iloc[-7:].mean()),
                "rolling_7d_std": float(series.iloc[-7:].std()),
                "rolling_14d_mean": float(series.iloc[-14:].mean()),
                "hidroeolica_lag1": float(hist["ree_hidroeolica_value"].dropna().iloc[-1]) if "ree_hidroeolica_value" in hist else 0.0,
            }, "historical"
    if manual_history:
        return {key: float(value) if value is not None else 0.0 for key, value in manual_history.items()}, "manual"
    raise ValueError(f"No hay suficiente historico eolico antes de {fecha}.")


def _wind_weather(fecha: str, manual_weather: dict[str, Any] | None) -> dict[str, Any]:
    if manual_weather:
        return manual_weather
    df = renewable_dataset()
    date_value = pd.Timestamp(fecha)
    row = df[df["date"] == date_value]
    if row.empty:
        try:
            return forecast_for_canarias(fecha)
        except Exception as exc:
            raise ValueError(
                f"No hay meteorologia agregada historica para Canarias en {fecha} y no se pudo consultar Open-Meteo: {exc}. "
                "Introduce un escenario meteorologico manual."
            ) from exc
    return _weather_from_row(row.iloc[0], municipal=False)


def _weather_from_row(row: pd.Series, municipal: bool) -> dict[str, Any]:
    return {
        "temp_avg_c": float(row["temp_avg_c"]),
        "temp_max_c": float(row["temp_max_c"]),
        "temp_min_c": float(row["temp_min_c"]),
        "humidity_avg_pct": float(row["humidity_avg_pct"]),
        "dew_point_avg_c": _optional_float(row.get("dew_point_avg_c")),
        "pressure_avg_hpa": _optional_float(row.get("pressure_avg_hpa")),
        "precip_intensity_avg_mm": _optional_float(row.get("precip_intensity_avg_mm")) or 0.0,
        "rain_daily_mm": _optional_float(row.get("rain_daily_mm")) or 0.0,
        "wind_speed_avg_ms": _optional_float(row.get("wind_speed_avg_ms")),
        "wind_speed_max_ms": _optional_float(row.get("wind_speed_max_ms")),
        "wind_speed_sdev_ms": _optional_float(row.get("wind_speed_sdev_ms")),
        "wind_dir_avg_deg": _optional_float(row.get("wind_dir_avg_deg")),
        "wind_dir_max_deg": _optional_float(row.get("wind_dir_max_deg")),
        "wind_dir_sdev_deg": _optional_float(row.get("wind_dir_sdev_deg")),
        "weather_station_count": int(row.get("weather_station_count", 1)) if municipal else int(row.get("canarias_weather_station_count", 1)),
    }


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _predict_wind_raw(payload: dict[str, Any]) -> float:
    features = build_wind_features(payload)
    bundle = _models["eolica"]
    values = bundle["imputer"].transform(features.to_numpy(dtype=float))
    pred = float(bundle["model"].predict(values)[0])
    return round(max(pred, 0.0), 4)


def _wind_sensitivity(payload: dict[str, Any]) -> list[dict[str, float]]:
    wind = payload["weather"].get("wind_speed_avg_ms")
    if wind is None:
        return []
    points = []
    for candidate in np.linspace(max(0.0, wind - 3.0), wind + 3.0, 7):
        pred = _predict_wind_raw(with_wind_speed(payload, float(candidate)))
        points.append({"label": f"{candidate:.1f} m/s", "value": pred})
    return points


def _wind_uncertainty(prediction: float, fecha: str) -> dict[str, Any]:
    metrics = _best_wind_metric()
    mae = float(metrics.get("MAE") or 900.0)
    wmape = float(metrics.get("WMAPE") or 22.5)
    half_width = max(mae, prediction * wmape / 100)
    date_value = pd.Timestamp(fecha)
    known_max = renewable_dataset()["date"].max()
    confidence = "media"
    if half_width / max(prediction, 1.0) > 0.35 or date_value > known_max:
        confidence = "baja"
    elif half_width / max(prediction, 1.0) < 0.2:
        confidence = "alta"
    return {
        "low": round(max(prediction - half_width, 0.0), 2),
        "high": round(prediction + half_width, 2),
        "confidence": confidence,
    }


def _best_wind_metric() -> dict[str, Any]:
    path = EVAL_DIR / "renewable_energy_generation" / "metricas_eolica.csv"
    if not path.exists():
        return {}
    rows = pd.read_csv(path)
    if rows.empty:
        return {}
    if "modelo" in rows.columns and "HGB" in rows["modelo"].values:
        return rows[rows["modelo"] == "HGB"].iloc[0].to_dict()
    return rows.iloc[0].to_dict()


def _wind_condition(wind_speed: float | None, prediction: float, rolling_7d: float) -> str:
    if wind_speed is not None and wind_speed >= 8.0:
        return "favorable"
    if prediction >= rolling_7d * 1.1:
        return "favorable"
    if prediction <= rolling_7d * 0.85:
        return "baja"
    return "normal"


def _read_csv_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_csv(path).replace({np.nan: None}).to_dict(orient="records")


def _status_for(keys: list[str]) -> dict[str, str]:
    return {key: "cargado" if key in _models else _model_errors.get(key, "no cargado") for key in keys}
