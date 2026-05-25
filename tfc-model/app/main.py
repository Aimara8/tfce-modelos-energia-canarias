from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Ensure tfc-model root is on sys.path so `import app.*` works when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.inference import (
    CONSUMPTION_FEATURES,
    CONSUMPTION_SECTORS,
    ConsumptionPredictionRequest,
    RenewablePredictionRequest,
    get_municipalities,
    get_renewable_feature_order,
    predict_consumption,
    predict_renewable,
)


STATIC_DIR = Path(__file__).resolve().parent / "static"
DASHBOARD_CONFIG_PATH = Path(__file__).resolve().parent / "dashboard_config.json"


def load_dashboards() -> list[dict[str, str]]:
    if not DASHBOARD_CONFIG_PATH.exists():
        return []

    try:
        dashboards = json.loads(DASHBOARD_CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    if not isinstance(dashboards, list):
        return []

    cleaned: list[dict[str, str]] = []
    for item in dashboards:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        provider = str(item.get("provider", "")).strip().lower()
        description = str(item.get("description", "")).strip()
        embed_url = str(item.get("embed_url", "")).strip()
        if title and embed_url:
            cleaned.append(
                {
                    "title": title,
                    "provider": provider or "dashboard",
                    "description": description,
                    "embed_url": embed_url,
                }
            )
    return cleaned

app = FastAPI(
    title="TFC Energy Forecast API",
    version="0.1.0",
    description="API local para inferencia de consumo electrico y generacion eolica.",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/metadata")
def metadata() -> dict[str, object]:
    return {
        "consumption": {
            "sectors": list(CONSUMPTION_SECTORS.keys()),
            "municipalities": get_municipalities(),
            "feature_count": len(CONSUMPTION_FEATURES),
        },
        "renewable": {
            "models": ["hgb", "ridge"],
            "feature_count": len(get_renewable_feature_order()),
        },
        "dashboards": load_dashboards(),
    }


@app.post("/api/predict/consumption")
def api_predict_consumption(payload: ConsumptionPredictionRequest) -> dict[str, object]:
    try:
        return predict_consumption(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/predict/eolica")
def api_predict_eolica(payload: RenewablePredictionRequest) -> dict[str, object]:
    try:
        return predict_renewable(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
