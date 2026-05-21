from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .feature_engineering import (
    CONSUMPTION_FEATURES,
    WIND_FEATURES,
    build_consumption_features,
    build_wind_features,
    with_wind_speed,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "src" / "models"

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
    except Exception as exc:  # pragma: no cover - depends on runtime environment
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
        if getattr(model, "n_features_in_", len(CONSUMPTION_FEATURES)) != len(CONSUMPTION_FEATURES):
            _model_errors[key] = "numero de features incompatible"
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


def predict_consumption(payload: dict[str, Any]) -> dict[str, Any]:
    ensure_models_loaded()
    features = build_consumption_features(payload)
    values = features.to_numpy(dtype=float)
    predictions = []

    for sector in ["total", "residencial", "servicios", "industria"]:
        key = f"consumo_{sector}"
        if key not in _models:
            predictions.append({"sector": sector, "mwh": None})
            continue
        pred = float(_models[key].predict(values)[0])
        predictions.append({"sector": sector, "mwh": round(max(pred, 0.0), 4)})

    history = payload["history"]
    reference = [
        {"label": "lag_1d", "value": history["lag_1d"]},
        {"label": "rolling_7d", "value": history["rolling_7d_mean"]},
        {"label": "rolling_30d", "value": history["rolling_30d_mean"]},
    ]
    return {
        "predictions": predictions,
        "chart_bars": [{"label": item["sector"], "value": item["mwh"]} for item in predictions],
        "chart_reference": reference,
        "model_status": _status_for(["consumo_total", "consumo_residencial", "consumo_servicios", "consumo_industria"]),
    }


def predict_wind(payload: dict[str, Any]) -> dict[str, Any]:
    ensure_models_loaded()
    history = payload["history"]
    chart_series = [
        {"label": "hace 3 dias", "value": history["lag_3d"]},
        {"label": "hace 2 dias", "value": history["lag_2d"]},
        {"label": "ayer", "value": history["lag_1d"]},
    ]

    if "eolica" not in _models:
        return {
            "eolica_predicha_mwh": None,
            "condition": "sin_modelo",
            "chart_series": chart_series + [{"label": "prediccion", "value": None}],
            "sensitivity_by_wind": [],
            "model_status": _status_for(["eolica"]),
        }

    pred = _predict_wind_raw(payload)
    chart_series.append({"label": "prediccion", "value": pred})
    sensitivity = _wind_sensitivity(payload)
    return {
        "eolica_predicha_mwh": pred,
        "condition": _wind_condition(payload["weather"].get("wind_speed_avg_ms"), pred, history["rolling_7d_mean"]),
        "chart_series": chart_series,
        "sensitivity_by_wind": sensitivity,
        "model_status": _status_for(["eolica"]),
    }


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


def _wind_condition(wind_speed: float | None, prediction: float, rolling_7d: float) -> str:
    if wind_speed is not None and wind_speed >= 8.0:
        return "favorable"
    if prediction >= rolling_7d * 1.1:
        return "favorable"
    if prediction <= rolling_7d * 0.85:
        return "baja"
    return "normal"


def _status_for(keys: list[str]) -> dict[str, str]:
    return {key: "cargado" if key in _models else _model_errors.get(key, "no cargado") for key in keys}

