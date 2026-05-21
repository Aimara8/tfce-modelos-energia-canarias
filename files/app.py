"""
client/app.py — Aplicación cliente Streamlit para la API de predicción
energética de Canarias.

Arrancar con:
    streamlit run client/app.py
"""

import requests
from datetime import date, timedelta
import streamlit as st

# ── Configuración ─────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Predicción Energética — Canarias",
    page_icon="⚡",
    layout="wide",
)

# ── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: #0f1117;
        color: #e8ecf0;
    }
    .metric-card {
        background: #1c2333;
        border: 1px solid #2d3a50;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card .label {
        font-size: 0.78rem;
        color: #7a8fa8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
    }
    .metric-card .value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #4fc3f7;
    }
    .metric-card .unit {
        font-size: 0.85rem;
        color: #7a8fa8;
    }
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #7a8fa8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 1.5rem 0 0.8rem;
        border-bottom: 1px solid #2d3a50;
        padding-bottom: 0.4rem;
    }
    div[data-testid="stTabs"] button {
        font-size: 0.95rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ⚡ Predicción Energética — Canarias")
st.markdown("Modelos predictivos de demanda eléctrica municipal y generación eólica diaria.")

# Estado de la API
with st.sidebar:
    st.markdown("### 🔌 Estado de la API")
    api_url_input = st.text_input("URL de la API", value=API_URL)
    if st.button("Verificar conexión"):
        try:
            r = requests.get(f"{api_url_input}/health", timeout=3)
            data = r.json()
            st.success(f"API activa ✓")
            st.info(f"Modelos cargados: {', '.join(data.get('modelos_cargados', []))}")
        except Exception as e:
            st.error(f"No se puede conectar: {e}")

    st.markdown("---")
    st.markdown("### ℹ️ Modelos")
    st.markdown("**Consumo:** XGBoost\n\nR² total = 0.9986")
    st.markdown("**Eólica:** HistGradientBoosting\n\nR² = 0.8168")
    st.markdown("---")
    st.caption("TFC — CIFP César Manrique · 2025")


# ── Tabs principales ──────────────────────────────────────────────────────────
tab_consumo, tab_eolica = st.tabs(["🏘️ Demanda Eléctrica Municipal", "💨 Generación Eólica"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CONSUMO ELÉCTRICO
# ══════════════════════════════════════════════════════════════════════════════
with tab_consumo:
    st.markdown("Introduce los datos del municipio y la meteorología para obtener la predicción de consumo.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">📍 Municipio y fecha</div>', unsafe_allow_html=True)
        municipio_cod = st.number_input(
            "Código INE del municipio",
            min_value=35001, max_value=38999,
            value=35016,
            help="Ej: 35016 = Las Palmas de Gran Canaria, 38038 = Santa Cruz de Tenerife"
        )
        fecha_consumo = st.date_input(
            "Fecha de predicción",
            value=date.today() + timedelta(days=1),
            key="fecha_consumo"
        )

        st.markdown('<div class="section-title">🌡️ Meteorología</div>', unsafe_allow_html=True)
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            temp_mean = st.number_input("T media (°C)", value=22.0, step=0.5)
        with col_t2:
            temp_max  = st.number_input("T máx (°C)", value=27.0, step=0.5)
        with col_t3:
            temp_min  = st.number_input("T mín (°C)", value=17.0, step=0.5)

        col_h1, col_h2, col_h3 = st.columns(3)
        with col_h1:
            humidity   = st.number_input("Humedad (%)", value=65.0, step=1.0)
        with col_h2:
            wind_speed = st.number_input("Viento (m/s)", value=3.5, step=0.1)
        with col_h3:
            precip     = st.number_input("Precipitación (mm)", value=0.0, step=0.1)

    with col_b:
        st.markdown('<div class="section-title">📅 Lags de consumo total (MWh)</div>', unsafe_allow_html=True)
        st.caption("Introduce los valores históricos de consumo total del municipio.")

        lag1 = st.number_input("Consumo ayer (lag 1d)", value=450.0, step=1.0)
        lag7 = st.number_input("Consumo hace 7 días (lag 7d)", value=445.0, step=1.0)
        lag14 = st.number_input("Consumo hace 14 días (lag 14d)", value=447.0, step=1.0)
        lag28 = st.number_input("Consumo hace 28 días (lag 28d)", value=442.0, step=1.0)
        roll7 = st.number_input("Media móvil 7 días", value=446.0, step=1.0)

    st.markdown("---")

    if st.button("🔮 Predecir consumo", type="primary", use_container_width=True):
        payload = {
            "municipio_cod": int(municipio_cod),
            "fecha": str(fecha_consumo),
            "temp_mean": temp_mean,
            "temp_max": temp_max,
            "temp_min": temp_min,
            "humidity_mean": humidity,
            "wind_speed_mean": wind_speed,
            "precipitation": precip,
            "demand_total_lag1": lag1,
            "demand_total_lag7": lag7,
            "demand_total_lag14": lag14,
            "demand_total_lag28": lag28,
            "demand_total_rolling7": roll7,
        }
        try:
            with st.spinner("Consultando el modelo..."):
                r = requests.post(f"{api_url_input}/predict/consumo", json=payload, timeout=10)
                r.raise_for_status()
                res = r.json()

            st.success(f"✅ Predicción para municipio {res['municipio_cod']} — {res['fecha']}")

            c1, c2, c3, c4 = st.columns(4)
            for col, sector, label, color in [
                (c1, "demand_total_mwh",       "Total",       "#4fc3f7"),
                (c2, "demand_residencial_mwh",  "Residencial", "#81c784"),
                (c3, "demand_servicios_mwh",    "Servicios",   "#ffb74d"),
                (c4, "demand_industria_mwh",    "Industria",   "#e57373"),
            ]:
                val = res.get(sector)
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="label">{label}</div>
                        <div class="value" style="color:{color}">{val:.1f}</div>
                        <div class="unit">MWh</div>
                    </div>
                    """, unsafe_allow_html=True)

            with st.expander("Ver respuesta completa (JSON)"):
                st.json(res)

        except requests.exceptions.ConnectionError:
            st.error("❌ No se puede conectar a la API. ¿Está arrancada en localhost:8000?")
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ Error de la API: {e.response.text}")
        except Exception as e:
            st.error(f"❌ Error inesperado: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GENERACIÓN EÓLICA
# ══════════════════════════════════════════════════════════════════════════════
with tab_eolica:
    st.markdown("Introduce la meteorología agregada del archipiélago y los lags de generación reciente.")

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.markdown('<div class="section-title">📅 Fecha</div>', unsafe_allow_html=True)
        fecha_eolica = st.date_input(
            "Fecha de predicción",
            value=date.today() + timedelta(days=1),
            key="fecha_eolica"
        )

        st.markdown('<div class="section-title">💨 Meteorología del archipiélago</div>', unsafe_allow_html=True)
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            wind_mean = st.number_input("Viento medio (m/s)", value=7.5, step=0.1, key="wm")
        with col_w2:
            wind_max_e = st.number_input("Viento máx (m/s)", value=12.0, step=0.1, key="wx")
        with col_w3:
            wind_std_e = st.number_input("Desv. estándar viento", value=2.3, step=0.1, key="ws")

        col_te1, col_te2 = st.columns(2)
        with col_te1:
            temp_e    = st.number_input("T media (°C)", value=22.0, step=0.5, key="te")
        with col_te2:
            humid_e   = st.number_input("Humedad (%)", value=70.0, step=1.0, key="he")

    with col_e2:
        st.markdown('<div class="section-title">⚡ Lags de generación eólica (MWh)</div>', unsafe_allow_html=True)
        st.caption("Generación eólica real de los últimos días para el conjunto de Canarias.")

        e_lag1   = st.number_input("Generación ayer (lag 1d)",       value=3850.0, step=10.0)
        e_lag2   = st.number_input("Generación hace 2 días (lag 2d)", value=3720.0, step=10.0)
        e_lag3   = st.number_input("Generación hace 3 días (lag 3d)", value=4010.0, step=10.0)
        e_roll3  = st.number_input("Media móvil 3 días",              value=3860.0, step=10.0)

    st.markdown("---")

    if st.button("🔮 Predecir generación eólica", type="primary", use_container_width=True):
        payload_e = {
            "fecha": str(fecha_eolica),
            "wind_speed_mean": wind_mean,
            "wind_speed_max": wind_max_e,
            "wind_speed_std": wind_std_e,
            "temp_mean": temp_e,
            "humidity_mean": humid_e,
            "eolica_lag1": e_lag1,
            "eolica_lag2": e_lag2,
            "eolica_lag3": e_lag3,
            "eolica_rolling3": e_roll3,
        }
        try:
            with st.spinner("Consultando el modelo eólico..."):
                r = requests.post(f"{api_url_input}/predict/eolica", json=payload_e, timeout=10)
                r.raise_for_status()
                res_e = r.json()

            st.success(f"✅ Predicción para {res_e['fecha']} — Modelo: {res_e['modelo']}")

            col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
            with col_r2:
                st.markdown(f"""
                <div class="metric-card" style="margin: 1rem 0;">
                    <div class="label">Generación Eólica Prevista</div>
                    <div class="value" style="font-size:3rem; color:#80cbc4">
                        {res_e['eolica_predicha_mwh']:,.0f}
                    </div>
                    <div class="unit" style="font-size:1rem">MWh · {res_e['fecha']}</div>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("Ver respuesta completa (JSON)"):
                st.json(res_e)

        except requests.exceptions.ConnectionError:
            st.error("❌ No se puede conectar a la API. ¿Está arrancada en localhost:8000?")
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ Error de la API: {e.response.text}")
        except Exception as e:
            st.error(f"❌ Error inesperado: {e}")
