from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

API_URL = "https://tfce-modelos-energia-canarias.onrender.com"

COLORS = {
    "consumption": "#10b981",
    "wind": "#f59e0b",
    "wind_dark": "#059669",
    "danger": "#ef4444",
    "accent": "#6366f1",
}

TOTAL_CANARY_MUNICIPALITIES = 87

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

COORDS_CSV = Path(__file__).resolve().parents[3] / "tfc-datasets" / "outputs" / "municipality_coordinates_open_meteo.csv"
if COORDS_CSV.exists():
    _coords_df = pd.read_csv(COORDS_CSV)
    MUNICIPALITY_COORDS.update({
        row["municipality"]: (float(row["latitude"]), float(row["longitude"]))
        for _, row in _coords_df.dropna(subset=["municipality", "latitude", "longitude"]).iterrows()
    })

ISLAND_CENTERS = {
    "Lanzarote":     (29.05, -13.65),
    "Fuerteventura": (28.36, -14.10),
    "Gran Canaria":  (28.02, -15.60),
    "Tenerife":      (28.30, -16.62),
    "La Gomera":     (28.12, -17.24),
    "La Palma":      (28.66, -17.88),
    "El Hierro":     (27.75, -18.05),
}

st.set_page_config(
    page_title="Predicción Energética Canarias",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── helpers ────────────────────────────────────────────────────────────
def mwh(value: float | None) -> str:
    if value is None:
        return "sin dato"
    return f"{value:,.0f} MWh".replace(",", ".")

def pct(value: float | None) -> str:
    if value is None:
        return "sin dato"
    return f"{value:.1f}%"

def api_get(path: str) -> dict:
    r = requests.get(f"{api_url}{path}", timeout=15)
    r.raise_for_status()
    return r.json()

def api_post(path: str, payload: dict) -> dict:
    r = requests.post(f"{api_url}{path}", json=payload, timeout=20)
    if not r.ok:
        raise RuntimeError(r.json().get("detail", r.text))
    return r.json()

@st.cache_data(ttl=30)
def load_metadata(base_url: str) -> dict:
    r = requests.get(f"{base_url}/api/metadata", timeout=15)
    r.raise_for_status()
    return r.json()

# ─── Leaflet map ───────────────────────────────────────────────────────────
def leaflet_map(municipalities: list[str], selected: str | None = None, height: int = 400) -> None:
    markers_js = []
    for name in municipalities:
        coords = MUNICIPALITY_COORDS.get(name)
        if not coords:
            continue
        lat, lon = coords
        is_sel = name == selected
        color   = "#f59e0b" if is_sel else "#10b981"
        radius  = 11 if is_sel else 7
        opacity = 1.0 if is_sel else 0.75
        safe_name = name.replace("'", "\\'")
        markers_js.append(
            f"""L.circleMarker([{lat},{lon}],{{radius:{radius},fillColor:"{color}",
            color:"white",weight:2,opacity:1,fillOpacity:{opacity}}})
            .addTo(map).bindPopup("<b>{safe_name}</b>");"""
        )

    labels_js = []
    for island, (lat, lon) in ISLAND_CENTERS.items():
        labels_js.append(
            f"""L.marker([{lat},{lon}],{{icon:L.divIcon({{className:'',
            html:'<span style="font-size:10px;font-weight:700;color:#94a3b8;'
            +'text-shadow:0 1px 2px #000,0 -1px 2px #000;">{island}</span>',
            iconAnchor:[30,8]}})
            }}).addTo(map);"""
        )

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
      *{{margin:0;padding:0;box-sizing:border-box;}}
      body{{background:#080f1e;}}
      #map{{width:100%;height:{height}px;border-radius:10px;}}
      .leaflet-popup-content-wrapper{{background:#1e293b;color:#f1f5f9;
        border:1px solid #334155;border-radius:8px;}}
      .leaflet-popup-tip{{background:#1e293b;}}
      .leaflet-control-zoom a{{background:#1e293b!important;color:#f1f5f9!important;
        border-color:#334155!important;}}
    </style></head><body><div id="map"></div>
    <script>
      var map=L.map('map',{{center:[28.3,-15.8],zoom:7,zoomControl:true}});
      L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',{{
        attribution:'&copy; CARTO',subdomains:'abcd',maxZoom:19}}).addTo(map);
      {''.join(markers_js)}
      {''.join(labels_js)}
    </script></body></html>"""
    components.html(html, height=height + 2, scrolling=False)

# ─── chart helpers ──────────────────────────────────────────────────────────
CHART_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cbd5e1", family="'DM Sans', sans-serif"),
    margin=dict(l=8, r=8, t=36, b=8),
    xaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
    yaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
)

def _apply(fig: go.Figure, extra: dict | None = None) -> go.Figure:
    fig.update_layout(**{**CHART_BASE, **(extra or {})})
    return fig

def prediction_bar(result: dict) -> go.Figure:
    df = pd.DataFrame(result["predictions"])
    fig = px.bar(df, x="sector", y="mwh", color="sector",
                 text=df["mwh"].map(mwh),
                 color_discrete_sequence=["#10b981","#6366f1","#f59e0b","#ec4899"])
    fig.update_traces(textposition="outside", textfont_color="#f1f5f9")
    return _apply(fig, {"height": 320, "showlegend": False, "yaxis_title": "MWh", "xaxis_title": ""})

def reference_line(result: dict) -> go.Figure:
    df = pd.DataFrame(result["chart_reference"])
    fig = px.line(df, x="label", y="value", markers=True,
                  color_discrete_sequence=[COLORS["consumption"]])
    return _apply(fig, {"height": 320, "yaxis_title": "MWh", "xaxis_title": ""})

def wind_series_chart(result: dict) -> go.Figure:
    df = pd.DataFrame(result["chart_series"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["label"], y=df["value"], mode="lines+markers",
                             name="Serie", line=dict(color=COLORS["wind_dark"], width=2.5)))
    low, high = result["uncertainty_low_mwh"], result["uncertainty_high_mwh"]
    if low is not None and high is not None:
        fig.add_trace(go.Scatter(x=["prediccion","prediccion"], y=[low, high],
                                 mode="lines", name="Incertidumbre",
                                 line=dict(color=COLORS["wind"], width=10)))
    return _apply(fig, {"height": 320, "yaxis_title": "MWh", "xaxis_title": ""})

def sensitivity_chart(result: dict) -> go.Figure:
    df = pd.DataFrame(result["sensitivity_by_wind"])
    fig = px.line(df, x="label", y="value", markers=True,
                  color_discrete_sequence=[COLORS["wind"]])
    return _apply(fig, {"height": 320, "yaxis_title": "MWh", "xaxis_title": "Viento medio"})

def donut_chart(rows: list[dict], names: str, values: str) -> go.Figure:
    df = pd.DataFrame(rows)
    fig = px.pie(df, names=names, values=values, hole=0.60,
                 color_discrete_sequence=["#10b981","#6366f1","#f59e0b","#ec4899","#38bdf8"])
    fig.update_traces(textposition="inside", textinfo="percent+label",
                      textfont=dict(color="#fff", size=11))
    return _apply(fig, {"height": 310,
                         "margin": dict(l=10, r=10, t=20, b=10),
                         "legend": dict(orientation="h", font=dict(color="#cbd5e1"))})

def area_chart(df: pd.DataFrame, x: str, y: str, color: str) -> go.Figure:
    fig = px.area(df, x=x, y=y, color_discrete_sequence=[color])
    fig.update_traces(fill="tozeroy", line_width=2)
    return _apply(fig, {"height": 280, "yaxis_title": "MWh", "xaxis_title": ""})

# ─── weather form ──────────────────────────────────────────────────────────
def weather_form(prefix: str, wind_default: float) -> dict:
    st.caption("Meteorología diaria agregada")
    c1, c2, c3 = st.columns(3)
    with c1:
        temp_avg = st.number_input("Temp. media (°C)", value=23.0, step=0.5, key=f"{prefix}_temp_avg")
        humidity = st.number_input("Humedad (%)", value=66.0, min_value=0.0, max_value=100.0, step=1.0, key=f"{prefix}_humidity")
        pressure = st.number_input("Presión (hPa)", value=1015.0, step=1.0, key=f"{prefix}_pressure")
    with c2:
        temp_max = st.number_input("Temp. máxima (°C)", value=27.0, step=0.5, key=f"{prefix}_temp_max")
        rain     = st.number_input("Lluvia diaria (mm)", value=0.0, min_value=0.0, step=0.1, key=f"{prefix}_rain")
        precip   = st.number_input("Intensidad precip. (mm)", value=0.0, min_value=0.0, step=0.1, key=f"{prefix}_precip")
    with c3:
        temp_min = st.number_input("Temp. mínima (°C)", value=19.0, step=0.5, key=f"{prefix}_temp_min")
        wind_avg = st.number_input("Viento medio (m/s)", value=wind_default, min_value=0.0, step=0.1, key=f"{prefix}_wind_avg")
        wind_max = st.number_input("Viento máximo (m/s)", value=max(wind_default+4.0, wind_default), min_value=0.0, step=0.1, key=f"{prefix}_wind_max")
    with st.expander("Variables avanzadas", expanded=False):
        a1, a2, a3 = st.columns(3)
        with a1:
            wind_std = st.number_input("Desv. viento (m/s)", value=1.5, min_value=0.0, step=0.1, key=f"{prefix}_wind_std")
        with a2:
            wind_dir = st.number_input("Dir. viento media (°)", value=180.0, min_value=0.0, max_value=360.0, step=1.0, key=f"{prefix}_wind_dir")
        with a3:
            wind_dir_std = st.number_input("Desv. dirección (°)", value=35.0, min_value=0.0, max_value=180.0, step=1.0, key=f"{prefix}_wind_dir_std")
        b1, b2 = st.columns(2)
        with b1:
            wind_dir_max = st.number_input("Dir. viento máxima (°)", value=220.0, min_value=0.0, max_value=360.0, step=1.0, key=f"{prefix}_wind_dir_max")
        with b2:
            stations = st.number_input("Estaciones usadas", value=3, min_value=1, step=1, key=f"{prefix}_stations")
    return {
        "temp_avg_c": temp_avg, "temp_max_c": temp_max, "temp_min_c": temp_min,
        "humidity_avg_pct": humidity, "pressure_avg_hpa": pressure,
        "precip_intensity_avg_mm": precip, "rain_daily_mm": rain,
        "wind_speed_avg_ms": wind_avg, "wind_speed_max_ms": wind_max,
        "wind_speed_sdev_ms": wind_std, "wind_dir_avg_deg": wind_dir,
        "wind_dir_max_deg": wind_dir_max, "wind_dir_sdev_deg": wind_dir_std,
        "weather_station_count": int(stations),
    }

# ─── CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }

.stApp { background: #080f1e; }

.block-container { padding: 1.5rem 2rem 2rem 2rem !important; max-width: 1400px; }

/* ── sidebar: clean dark, no color fights ── */
section[data-testid="stSidebar"] {
  background: #0d1117 !important;
  border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * { color: #8b949e !important; }
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] strong { color: #c9d1d9 !important; }
/* active radio option */
section[data-testid="stSidebar"] .stRadio [aria-checked="true"] + div { color: #f0f6fc !important; }
section[data-testid="stSidebar"] input {
  background: #161b22 !important;
  border: 1px solid #30363d !important;
  color: #c9d1d9 !important;
  border-radius: 6px !important;
}
section[data-testid="stSidebar"] .stSuccess { background: #0d2818 !important; }
section[data-testid="stSidebar"] .stSuccess * { color: #3fb950 !important; }

/* ── metrics ── */
div[data-testid="stMetric"] {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 10px;
  padding: 14px 18px;
  transition: border-color .18s;
}
div[data-testid="stMetric"]:hover { border-color: #10b981; }
div[data-testid="stMetric"] label {
  color: #484f58 !important;
  font-size: 0.74rem !important;
  font-weight: 600 !important;
  text-transform: uppercase;
  letter-spacing: .07em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color: #e6edf3 !important; font-weight: 600 !important; font-size: 1.35rem !important;
}
div[data-testid="stMetricDelta"] { color: #3fb950 !important; }

/* ── page title ── */
.page-title { font-size: 1.6rem; font-weight: 600; color: #e6edf3; letter-spacing: -.02em; }
.page-subtitle { font-size: 0.85rem; color: #484f58; margin: .15rem 0 1.4rem 0; }

/* ── section label ── */
.slabel {
  font-size: .7rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: .1em; color: #10b981; margin-bottom: .45rem;
}

/* ── badge ── */
.badge {
  display: inline-block; background: #0d2818;
  border: 1px solid #1a4731; border-radius: 999px;
  padding: 2px 10px; font-size: .78rem; color: #3fb950;
  font-weight: 500; margin: 2px 2px 0 0;
}

hr { border-color: #21262d !important; margin: 1rem 0 !important; }

/* ── inputs ── */
.stSelectbox>div>div, .stDateInput>div>div, .stNumberInput>div>div {
  background: #161b22 !important; border-color: #30363d !important;
  color: #c9d1d9 !important; border-radius: 8px !important;
}

/* ── primary button ── */
.stButton>button[kind="primary"] {
  background: linear-gradient(135deg,#10b981,#059669) !important;
  border: none !important; color: #fff !important;
  font-weight: 600 !important; border-radius: 8px !important;
  padding: .55rem 1.5rem !important;
  transition: opacity .15s, transform .1s !important;
}
.stButton>button[kind="primary"]:hover { opacity: .88 !important; transform: translateY(-1px) !important; }

/* ── alerts ── */
.stAlert { border-radius: 8px !important; border-left-width: 3px !important; }

/* ── plotly card wrapper ── */
[data-testid="stPlotlyChart"] {
  background: #0d1117; border: 1px solid #21262d;
  border-radius: 10px; padding: 4px;
}

/* ── expander ── */
details summary { color: #8b949e !important; }
</style>
""", unsafe_allow_html=True)


# ─── sidebar ──────────────────────────────────────────────────────────[[...]
with st.sidebar:
    st.markdown("**⚡ Demo TFC**")
    st.markdown('<div class="slabel" style="margin-top:.5rem">Secciones</div>', unsafe_allow_html=True)
    page = st.radio(
        "nav",
        ["🔮 Predicción consumo", "🌬️ Predicción eólica",
         "📊 Histórico consumo",  "📈 Histórico generación"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown('<div class="slabel">API</div>', unsafe_allow_html=True)
    api_url = st.text_input("url", value=API_URL, label_visibility="collapsed")
    try:
        metadata = load_metadata(api_url)
        health   = api_get("/api/health")
        st.success("API activa")
        for m in health.get("modelos_cargados", []):
            st.markdown(f'<span class="badge">{m}</span>', unsafe_allow_html=True)
    except Exception as exc:
        st.error(f"Sin conexión: {exc}")
        st.stop()


# ═════════════════════════════════════════════════════════════════
# PAGE: Histórico consumo
# ═════════════════════════════════════════════════════════════════
if page == "📊 Histórico consumo":
    st.markdown('<span class="page-title">📊 Histórico de consumo</span>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Análisis histórico de demanda eléctrica por municipio</div>', unsafe_allow_html=True)

    consumo_dash = api_get("/api/dashboard/consumo")
    k = consumo_dash.get("kpis", {})

    # intentar obtener consumo medio diario desde la API; si no existe, aproximar desde monthly_total
    avg_daily = k.get("avg_daily_consumption")
    if avg_daily is None and consumo_dash.get("monthly_total"):
        try:
            _mdf = pd.DataFrame(consumo_dash["monthly_total"])
            if "mwh" in _mdf.columns and not _mdf["mwh"].empty:
                avg_daily = float(_mdf["mwh"].mean()) / 30.0  # aproximación diaria
        except Exception:
            avg_daily = None

    # KPIs contextuales de esta sección
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Municipios modelados", f"{k.get('municipalities')}/{TOTAL_CANARY_MUNICIPALITIES}")
    c2.metric("Consumo medio diario", mwh(avg_daily))
    c3.metric("Último día disponible", k.get("date_max"))
    c4.metric("Demanda último día", mwh(k.get("latest_total_mwh")))

    st.markdown("---")
    m1, m2 = st.columns([1.35, 1])
    with m1:
        st.markdown('<div class="slabel">Distribución geográfica</div>', unsafe_allow_html=True)
        leaflet_map(metadata["consumption"]["municipalities"], height=390)
    with m2:
        st.markdown('<div class="slabel">Mix de consumo · último día</div>', unsafe_allow_html=True)
        st.plotly_chart(donut_chart(consumo_dash["sector_totals"], "sector", "mwh"), use_container_width=True)

    st.markdown("---")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown('<div class="slabel">Demanda total mensual</div>', unsafe_allow_html=True)
        monthly_df = pd.DataFrame(consumo_dash["monthly_total"])
        st.plotly_chart(area_chart(monthly_df, "month", "mwh", COLORS["consumption"]), use_container_width=True)
    with g2:
        st.markdown('<div class="slabel">Top municipios · último día</div>', unsafe_allow_html=True)
        top_df = pd.DataFrame(consumo_dash["top_municipalities"])
        fig_top = px.bar(top_df, x="mwh", y="municipality", orientation="h",
                         color_discrete_sequence=[COLORS["accent"]])
        st.plotly_chart(_apply(fig_top, {"height": 280}), use_container_width=True)


# ═════════════════════════════════════════════════════════════════
# PAGE: Predicción consumo
# ═════════════════════════════════���═══════════════════════════════
elif page == "🔮 Predicción consumo":
    st.markdown('<span class="page-title">🔮 Predicción de consumo</span>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Estimación de demanda eléctrica diaria por municipio y sector</div>', unsafe_allow_html=True)

    # ── configuración + mapa en la misma fila ──────────────────────────────────
    cfg_col, map_col = st.columns([1, 1.4])
    with cfg_col:
        st.markdown('<div class="slabel">Escenario</div>', unsafe_allow_html=True)
        municipality = st.selectbox("Municipio", metadata["consumption"]["municipalities"])
        try:
            default_date = pd.to_datetime(metadata.get("consumption", {}).get("date_max", pd.Timestamp.now())).date()
        except (KeyError, TypeError, ValueError):
            default_date = pd.Timestamp.now().date()
        fecha = st.date_input("Fecha de predicción", value=default_date, key="fecha_consumo")
        use_hist = st.checkbox("Meteorología automática", value=True, key="consumo_hist_weather")
        st.info("Usa histórico local si existe. Para fechas futuras intenta Open-Meteo.")
        if not use_hist:
            st.markdown('<div class="slabel" style="margin-top:.8rem">Parámetros meteorológicos</div>', unsafe_allow_html=True)
            weather = weather_form("consumo", wind_default=5.0)
        else:
            weather = None
    with map_col:
        st.markdown('<div class="slabel">Municipio seleccionado</div>', unsafe_allow_html=True)
        leaflet_map(metadata["consumption"]["municipalities"], selected=municipality, height=420)

    st.markdown("---")
    if st.button("Predecir consumo", type="primary", use_container_width=True):
        try:
            payload = {"municipality": municipality, "fecha": str(fecha)}
            if weather:
                payload["weather"] = weather
            result = api_post("/api/predict/consumption", payload)
            if result["warnings"]:
                st.warning("⚠️ " + " ".join(result["warnings"]))

            st.markdown('<div class="slabel">Resultados</div>', unsafe_allow_html=True)
            mc = st.columns(4)
            for col, item in zip(mc, result["predictions"]):
                val = item.get("mwh")
                baseline = item.get("baseline_mwh")
                delta_num = None
                try:
                    if isinstance(val, (int, float)) and isinstance(baseline, (int, float)):
                        delta_num = val - baseline
                except Exception:
                    delta_num = None
                # pasar delta_num (numérico) para que Streamlit coloree el cambio; mostrar baseline en help
                col.metric(item["sector"].capitalize(), mwh(val), delta=delta_num if delta_num is not None else None,
                           help=f"Baseline: {mwh(baseline)}")

            g1, g2 = st.columns([1.2, 1])
            with g1:
                st.markdown('<div class="slabel">Desglose por sector</div>', unsafe_allow_html=True)
                st.plotly_chart(prediction_bar(result), use_container_width=True)
            with g2:
                st.markdown('<div class="slabel">Referencia histórica</div>', unsafe_allow_html=True)
                st.plotly_chart(reference_line(result), use_container_width=True)
        except Exception as exc:
            st.error(f"Error: {exc}")


# ═════════════════════════════════════════════════════════════════
# PAGE: Histórico generación
# ═════════════════════════════════════════════════════════════════
elif page == "📈 Histórico generación":
    st.markdown('<span class="page-title">📈 Histórico de generación</span>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Evolución de la generación eólica y renovable en las Islas Canarias</div>', unsafe_allow_html=True)

    eolica_dash = api_get("/api/dashboard/eolica")
    k = eolica_dash.get("kpis", {})
    best = next((r for r in eolica_dash["metrics"] if r.get("modelo") == "HGB"), {})

    # intentar calcular % cobertura eólica sobre el total (si viene latest_mix)
    coverage_pct = k.get("renewable_coverage_pct")
    if coverage_pct is None and eolica_dash.get("latest_mix"):
        try:
            _mix = pd.DataFrame(eolica_dash["latest_mix"])
            if {"technology", "mwh"}.issubset(_mix.columns):
                total = _mix["mwh"].sum() if not _mix["mwh"].empty else 0.0
                eolica_sum = _mix.loc[_mix["technology"].str.lower().str.contains("eol", na=False), "mwh"].sum()
                coverage_pct = (eolica_sum / total * 100.0) if total > 0 else None
        except Exception:
            coverage_pct = None

    # KPIs contextuales de esta sección
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Eólica último día", mwh(k.get("latest_eolica_mwh")))
    e2.metric("Viento medio", f"{k.get('latest_wind_ms', 0.0):.2f} m/s")
    e3.metric("% cobertura renovable", pct(coverage_pct))
    e4.metric("Precisión modelo (WMAPE)", pct(best.get("WMAPE")))

    st.markdown("---")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown('<div class="slabel">Mix renovable · último día</div>', unsafe_allow_html=True)
        st.plotly_chart(donut_chart(eolica_dash["latest_mix"], "technology", "mwh"), use_container_width=True)
    with g2:
        monthly = pd.DataFrame(eolica_dash["monthly"])
        fig_gen = go.Figure()
        fig_gen.add_trace(go.Scatter(x=monthly["month"], y=monthly["eolica_mwh"],
                                      mode="lines+markers", name="Eólica",
                                      line=dict(color=COLORS["wind_dark"], width=2.5)))
        fig_gen.add_trace(go.Scatter(x=monthly["month"], y=monthly["solar_mwh"],
                                      mode="lines+markers", name="Solar",
                                      line=dict(color=COLORS["wind"], width=2.5, dash="dot")))
        st.markdown('<div class="slabel">Generación media mensual</div>', unsafe_allow_html=True)
        st.plotly_chart(_apply(fig_gen, {"height": 310,
            "legend": dict(orientation="h", font=dict(color="#8b949e"))}), use_container_width=True)

    st.markdown("---")
    g3, g4 = st.columns(2)
    with g3:
        err_df = pd.DataFrame(eolica_dash["monthly_errors"])
        fig_err = px.bar(err_df, x="month", y="wmape", color_discrete_sequence=[COLORS["danger"]])
        st.markdown('<div class="slabel">Error WMAPE mensual</div>', unsafe_allow_html=True)
        st.plotly_chart(_apply(fig_err, {"height": 270}), use_container_width=True)
    with g4:
        fig_wind = px.line(monthly, x="month", y="wind_ms", markers=True,
                            color_discrete_sequence=[COLORS["accent"]])
        st.markdown('<div class="slabel">Viento medio mensual</div>', unsafe_allow_html=True)
        st.plotly_chart(_apply(fig_wind, {"height": 270}), use_container_width=True)


# ═════════════════════════════════════════════════════════════════
# PAGE: Predicción eólica
# ═════════════════════════════════════════════════════════════════
else:
    st.markdown('<span class="page-title">🌬️ Predicción eólica</span>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Estimación de generación eólica con intervalo de incertidumbre</div>', unsafe_allow_html=True)

    # ── configuración + mapa en la misma fila ──────────────────────────────────
    cfg_col, map_col = st.columns([1, 1.4])
    with cfg_col:
        st.markdown('<div class="slabel">Escenario</div>', unsafe_allow_html=True)
        try:
            default_wind_date = pd.to_datetime(metadata.get("renewable", {}).get("date_max", pd.Timestamp.now())).date()
        except (KeyError, TypeError, ValueError):
            default_wind_date = pd.Timestamp.now().date()
        fecha_eolica  = st.date_input("Fecha", value=default_wind_date, key="fecha_eolica")
        mun_count     = st.number_input("Municipios con meteorología", value=87, min_value=1, step=1)
        station_count = st.number_input("Estaciones agregadas",        value=50, min_value=1, step=1)
        use_hist_wind = st.checkbox("Meteorología automática agregada", value=True, key="eolica_hist_weather")
        st.warning("La predicción incluye un rango de incertidumbre, no es un valor puntual exacto.")
        if not use_hist_wind:
            st.markdown('<div class="slabel" style="margin-top:.8rem">Parámetros meteorológicos</div>', unsafe_allow_html=True)
            weather_e = weather_form("eolica", wind_default=7.5)
        else:
            weather_e = None
    with map_col:
        st.markdown('<div class="slabel">Cobertura de estaciones meteorológicas</div>', unsafe_allow_html=True)
        leaflet_map(metadata["consumption"]["municipalities"], height=420)

    st.markdown("---")
    if st.button("Predecir eólica", type="primary", use_container_width=True):
        try:
            payload_e = {
                "fecha": str(fecha_eolica),
                "canarias_weather_municipality_count": int(mun_count),
                "canarias_weather_station_count": int(station_count),
            }
            if weather_e:
                payload_e["weather"] = weather_e
            result_e = api_post("/api/predict/eolica", payload_e)
            if result_e["warnings"]:
                st.warning("⚠️ " + " ".join(result_e["warnings"]))

            st.markdown('<div class="slabel">Resultados</div>', unsafe_allow_html=True)
            ec1, ec2, ec3 = st.columns(3)
            ec1.metric("Predicción central", mwh(result_e.get("eolica_predicha_mwh")), result_e.get("condition"))
            ec2.metric("Rango estimado", f"{mwh(result_e.get('uncertainty_low_mwh'))} – {mwh(result_e.get('uncertainty_high_mwh'))}")
            # renombrado: mostrar comparación frente a la media 7 días; dejar 'confidence' como help
            ec3.metric("vs media 7 días", pct(result_e.get("comparison_to_rolling_7d_pct")), help=result_e.get("confidence", "").capitalize())

            g1, g2 = st.columns([1.2, 1])
            with g1:
                st.markdown('<div class="slabel">Serie histórica + predicción</div>', unsafe_allow_html=True)
                st.plotly_chart(wind_series_chart(result_e), use_container_width=True)
            if result_e.get("sensitivity_by_wind"):
                with g2:
                    st.markdown('<div class="slabel">Sensibilidad al viento</div>', unsafe_allow_html=True)
                    st.plotly_chart(sensitivity_chart(result_e), use_container_width=True)
        except Exception as exc:
            st.error(f"Error: {exc}")
