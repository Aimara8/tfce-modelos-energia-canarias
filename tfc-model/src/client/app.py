from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Prediccion Energetica Canarias", page_icon="EC", layout="wide")


def weather_payload(prefix: str, wind_default: float) -> dict:
    col1, col2, col3 = st.columns(3)
    with col1:
        temp_avg = st.number_input("Temperatura media (C)", value=22.0, step=0.5, key=f"{prefix}_temp_avg")
        humidity = st.number_input("Humedad media (%)", value=65.0, step=1.0, key=f"{prefix}_humidity")
        pressure = st.number_input("Presion media (hPa)", value=1015.0, step=1.0, key=f"{prefix}_pressure")
    with col2:
        temp_max = st.number_input("Temperatura maxima (C)", value=27.0, step=0.5, key=f"{prefix}_temp_max")
        rain = st.number_input("Lluvia diaria (mm)", value=0.0, step=0.1, key=f"{prefix}_rain")
        precip = st.number_input("Intensidad precip. (mm)", value=0.0, step=0.1, key=f"{prefix}_precip")
    with col3:
        temp_min = st.number_input("Temperatura minima (C)", value=17.0, step=0.5, key=f"{prefix}_temp_min")
        wind_avg = st.number_input("Viento medio (m/s)", value=wind_default, step=0.1, key=f"{prefix}_wind_avg")
        wind_max = st.number_input("Viento maximo (m/s)", value=max(wind_default + 3.0, wind_default), step=0.1, key=f"{prefix}_wind_max")

    return {
        "temp_avg_c": temp_avg,
        "temp_max_c": temp_max,
        "temp_min_c": temp_min,
        "humidity_avg_pct": humidity,
        "pressure_avg_hpa": pressure,
        "precip_intensity_avg_mm": precip,
        "rain_daily_mm": rain,
        "wind_speed_avg_ms": wind_avg,
        "wind_speed_max_ms": wind_max,
        "wind_speed_sdev_ms": st.sidebar.number_input(f"Desv. viento {prefix}", value=1.5, step=0.1),
        "wind_dir_avg_deg": st.sidebar.number_input(f"Dir. viento {prefix}", value=45.0, step=1.0),
        "wind_dir_max_deg": st.sidebar.number_input(f"Dir. max {prefix}", value=90.0, step=1.0),
        "wind_dir_sdev_deg": st.sidebar.number_input(f"Desv. dir {prefix}", value=20.0, step=1.0),
        "weather_station_count": st.sidebar.number_input(f"Estaciones {prefix}", value=3, min_value=1, step=1),
    }


def post_json(path: str, payload: dict) -> dict:
    response = requests.post(f"{api_url}{path}", json=payload, timeout=15)
    response.raise_for_status()
    return response.json()


st.title("Prediccion energetica Canarias")

with st.sidebar:
    api_url = st.text_input("API", value=API_URL)
    if st.button("Comprobar API"):
        try:
            health = requests.get(f"{api_url}/health", timeout=5).json()
            st.success("API activa")
            st.json(health)
        except Exception as exc:
            st.error(f"No se pudo conectar: {exc}")

tab_consumo, tab_eolica = st.tabs(["Consumo", "Eolica"])

with tab_consumo:
    left, right = st.columns([1, 2])
    with left:
        municipality_enc = st.number_input("Municipio interno", min_value=0, value=12, step=1)
        fecha = st.date_input("Fecha", value=date.today() + timedelta(days=1), key="fecha_consumo")
        lag_1d = st.number_input("Consumo ayer (MWh)", value=450.0, step=1.0)
        lag_7d = st.number_input("Consumo hace 7 dias (MWh)", value=443.0, step=1.0)
        lag_14d = st.number_input("Consumo hace 14 dias (MWh)", value=448.0, step=1.0)
        lag_28d = st.number_input("Consumo hace 28 dias (MWh)", value=441.0, step=1.0)
        rolling_7d = st.number_input("Media 7 dias", value=446.0, step=1.0)
        rolling_30d = st.number_input("Media 30 dias", value=444.0, step=1.0)
        rolling_std = st.number_input("Desv. 7 dias", value=8.0, step=0.5)
    with right:
        weather = weather_payload("consumo", wind_default=3.5)

    if st.button("Predecir consumo", type="primary", use_container_width=True):
        payload = {
            "municipality_enc": int(municipality_enc),
            "fecha": str(fecha),
            "weather": weather,
            "history": {
                "lag_1d": lag_1d,
                "lag_7d": lag_7d,
                "lag_14d": lag_14d,
                "lag_28d": lag_28d,
                "rolling_7d_mean": rolling_7d,
                "rolling_30d_mean": rolling_30d,
                "rolling_7d_std": rolling_std,
            },
        }
        try:
            result = post_json("/predict/consumo", payload)
            bars = pd.DataFrame(result["chart_bars"])
            refs = pd.DataFrame(result["chart_reference"])
            st.subheader("Prediccion por sector")
            st.bar_chart(bars, x="label", y="value")
            st.subheader("Referencia historica")
            st.line_chart(refs, x="label", y="value")
            st.json(result["model_status"])
        except Exception as exc:
            st.error(f"Error en prediccion: {exc}")

with tab_eolica:
    left, right = st.columns([1, 2])
    with left:
        fecha_eolica = st.date_input("Fecha", value=date.today() + timedelta(days=1), key="fecha_eolica")
        e_lag1 = st.number_input("Eolica ayer (MWh)", value=3850.0, step=10.0)
        e_lag2 = st.number_input("Eolica hace 2 dias (MWh)", value=3720.0, step=10.0)
        e_lag3 = st.number_input("Eolica hace 3 dias (MWh)", value=4010.0, step=10.0)
        e_roll3 = st.number_input("Media eolica 3 dias", value=3860.0, step=10.0)
        e_roll7 = st.number_input("Media eolica 7 dias", value=3800.0, step=10.0)
        e_std7 = st.number_input("Desv. eolica 7 dias", value=420.0, step=10.0)
        e_roll14 = st.number_input("Media eolica 14 dias", value=3750.0, step=10.0)
        hidroeolica = st.number_input("Hidroeolica ayer (MWh)", value=120.0, step=5.0)
    with right:
        weather_e = weather_payload("eolica", wind_default=7.5)

    if st.button("Predecir eolica", type="primary", use_container_width=True):
        payload_e = {
            "fecha": str(fecha_eolica),
            "weather": weather_e,
            "canarias_weather_municipality_count": 80,
            "canarias_weather_station_count": 100,
            "history": {
                "lag_1d": e_lag1,
                "lag_2d": e_lag2,
                "lag_3d": e_lag3,
                "rolling_3d_mean": e_roll3,
                "rolling_7d_mean": e_roll7,
                "rolling_7d_std": e_std7,
                "rolling_14d_mean": e_roll14,
                "hidroeolica_lag1": hidroeolica,
            },
        }
        try:
            result_e = post_json("/predict/eolica", payload_e)
            st.metric("Generacion eolica prevista", result_e["eolica_predicha_mwh"], result_e["condition"])
            series = pd.DataFrame(result_e["chart_series"])
            st.line_chart(series, x="label", y="value")
            sensitivity = pd.DataFrame(result_e["sensitivity_by_wind"])
            if not sensitivity.empty:
                st.subheader("Sensibilidad al viento")
                st.line_chart(sensitivity, x="label", y="value")
            st.json(result_e["model_status"])
        except Exception as exc:
            st.error(f"Error en prediccion: {exc}")
