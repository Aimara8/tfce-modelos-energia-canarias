"""
=============================================================
MODELO PREDICTIVO DE CONSUMO ELÉCTRICO - CANARIAS
TFC · XGBoost + RandomizedSearchCV + Multi-target sectorial
=============================================================
Targets modelados:
  · demand_total_mwh
  · demand_residencial_mwh
  · demand_servicios_mwh
  · demand_industria_mwh

Métricas: MAPE, R², MAE, RMSE
Optimización: RandomizedSearchCV con TimeSeriesSplit
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import randint, uniform

import xgboost as xgb

# ─────────────────────────────────────────────────────────────
# 0. CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────────────────────
# Rutas relativas robustas usando la carpeta raíz del proyecto
ROOT_DIR    = Path(__file__).resolve().parents[3]  # tfc-model
SRC_DIR     = ROOT_DIR / "src"
DATA_PATH   = ROOT_DIR / "data" / "final_demand_consumption_dataset.csv"

RANDOM_SEED = 42
TEST_RATIO  = 0.2

MODEL_DIR   = SRC_DIR / "models" / "consumption_energy_demand"
EVAL_DIR    = SRC_DIR / "evaluation" / "consumption_energy_demand"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Todos los targets que quieres modelar
SECTOR_TARGETS = {
    "total"      : "demand_total_mwh",
    "residencial": "demand_residencial_mwh",
    "servicios"  : "demand_servicios_mwh",
    "industria"  : "demand_industria_mwh",
}

BASE_TEMP = 18.0   # temperatura base para HDD/CDD

# ─────────────────────────────────────────────────────────────
# 1. CARGA Y LIMPIEZA
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("1. CARGA DE DATOS")
print("=" * 65)

df_raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
print(f"   Filas     : {len(df_raw):,}")
print(f"   Columnas  : {df_raw.shape[1]}")
print(f"   Fechas    : {df_raw['date'].min().date()} -> {df_raw['date'].max().date()}")
print(f"   Municipios: {df_raw['municipality'].nunique()}")

# ─────────────────────────────────────────────────────────────
# 2. INGENIERÍA DE FEATURES  (común a todos los targets)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("2. INGENIERÍA DE FEATURES")
print("=" * 65)

def build_features(df_in: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Construye todas las features para un target dado.
    Devuelve copia limpia lista para modelar.
    """
    df = df_in.dropna(subset=[target_col]).copy()

    # Calendario
    df["year"]       = df["date"].dt.year
    df["month"]      = df["date"].dt.month
    df["dayofweek"]  = df["date"].dt.dayofweek
    df["dayofyear"]  = df["date"].dt.dayofyear
    df["quarter"]    = df["date"].dt.quarter
    df["week"]       = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Estacionalidad cíclica (evita discontinuidad dic->ene)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]     / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]     / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dayofweek"] / 7)

    # Grados-día calefacción / refrigeración
    df["hdd"]          = np.maximum(BASE_TEMP - df["temp_avg_c"], 0)
    df["cdd"]          = np.maximum(df["temp_avg_c"] - BASE_TEMP, 0)
    df["temp_range_c"] = df["temp_max_c"] - df["temp_min_c"]

    # Codificación municipio
    le = LabelEncoder()
    df["municipality_enc"] = le.fit_transform(df["municipality"])

    # Lags y rolling – ordenar primero por municipio+fecha
    df = df.sort_values(["municipality", "date"]).reset_index(drop=True)

    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}d"] = df.groupby("municipality")[target_col].shift(lag)

    df["rolling_7d_mean"]  = (
        df.groupby("municipality")[target_col]
        .transform(lambda x: x.shift(1).rolling(7,  min_periods=3).mean())
    )
    df["rolling_30d_mean"] = (
        df.groupby("municipality")[target_col]
        .transform(lambda x: x.shift(1).rolling(30, min_periods=7).mean())
    )
    df["rolling_7d_std"] = (
        df.groupby("municipality")[target_col]
        .transform(lambda x: x.shift(1).rolling(7,  min_periods=3).std())
    )

    # Eliminar filas con lags incompletos
    lag_cols = (
        [f"lag_{l}d" for l in [1, 7, 14, 28]]
        + ["rolling_7d_mean", "rolling_30d_mean", "rolling_7d_std"]
    )
    df = df.dropna(subset=lag_cols).copy()
    return df


# Lista de features (igual para todos los modelos)
FEATURES = [
    "municipality_enc",
    # Calendario
    "year", "month", "dayofweek", "dayofyear", "quarter", "week", "is_weekend",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    # Meteorología
    "temp_avg_c", "temp_max_c", "temp_min_c", "temp_range_c",
    "humidity_avg_pct", "dew_point_avg_c",
    "pressure_avg_hpa",
    "precip_intensity_avg_mm", "rain_daily_mm",
    "wind_speed_avg_ms", "wind_speed_max_ms", "wind_speed_sdev_ms",
    "wind_dir_avg_deg",
    "weather_station_count",
    # Ingeniería térmica
    "hdd", "cdd",
    # Lags y rolling
    "lag_1d", "lag_7d", "lag_14d", "lag_28d",
    "rolling_7d_mean", "rolling_30d_mean", "rolling_7d_std",
]

# ─────────────────────────────────────────────────────────────
# 3. FUNCIONES DE MÉTRICAS
# ─────────────────────────────────────────────────────────────
def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

"""
El sector industrial presenta una distribución muy sesgada (mediana 3.18 MWh, 34% de registros < 1 MWh), lo que hace que el MAPE no sea una métrica apropiada. 
Se emplea WMAPE, que pondera el error por el volumen real y neutraliza el efecto de los valores cercanos a cero.
"""

def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100)

def compute_metrics(name: str, y_true, y_pred) -> dict:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    mae_v  = mean_absolute_error(yt, yp)
    rmse_v = np.sqrt(mean_squared_error(yt, yp))
    r2_v   = r2_score(yt, yp)
    mape_v = mape(yt, yp)
    wmape_v = wmape(yt, yp)
    
    print(f"      MAE   : {mae_v:>12.4f} MWh")
    print(f"      RMSE  : {rmse_v:>12.4f} MWh")
    print(f"      R2    : {r2_v:>12.4f}")
    print(f"      MAPE  : {mape_v:>11.2f} %")
    print(f"      WMAPE : {wmape_v:>11.2f} %")
    return {"split": name, "MAE": mae_v, "RMSE": rmse_v, "R2": r2_v, "MAPE": mape_v, "WMAPE": wmape_v}

# ─────────────────────────────────────────────────────────────
# 4. ESPACIO DE BÚSQUEDA (RandomizedSearch)
# ─────────────────────────────────────────────────────────────
PARAM_DIST = {
    "n_estimators"    : randint(400, 1200),
    "learning_rate"   : uniform(0.01, 0.14),   # 0.01 – 0.15
    "max_depth"       : randint(3, 9),
    "subsample"       : uniform(0.6, 0.4),     # 0.6 – 1.0
    "colsample_bytree": uniform(0.5, 0.5),     # 0.5 – 1.0
    "min_child_weight": randint(1, 15),
    "gamma"           : uniform(0.0, 0.5),
    "reg_alpha"       : uniform(0.0, 1.0),
    "reg_lambda"      : uniform(0.5, 2.0),
}

N_ITER    = 40   # combinaciones a probar; sube a 80-100 para el TFC final
CV_SPLITS = 5    # splits temporales

# ─────────────────────────────────────────────────────────────
# 5. BUCLE PRINCIPAL: un modelo por sector
# ─────────────────────────────────────────────────────────────
all_metrics    = []
trained_models = {}   # sector -> (model, feats, test_df, target_col)

for sector, target_col in SECTOR_TARGETS.items():

    print("\n" + "=" * 65)
    print(f"  SECTOR: {sector.upper()}  ->  {target_col}")
    print("=" * 65)

    # 5.1 Construir features
    df    = build_features(df_raw, target_col)
    feats = [f for f in FEATURES if f in df.columns]

    # 5.2 Split temporal estricto
    df_sorted = df.sort_values("date").reset_index(drop=True)
    cutoff    = int(len(df_sorted) * (1 - TEST_RATIO))
    train_df  = df_sorted.iloc[:cutoff]
    test_df   = df_sorted.iloc[cutoff:]

    X_train = train_df[feats].values
    y_train = train_df[target_col].values
    X_test  = test_df[feats].values
    y_test  = test_df[target_col].values

    print(f"  Train: {len(train_df):,} filas | Test: {len(test_df):,} filas")

    # 5.3 RandomizedSearchCV con TimeSeriesSplit
    print(f"\n  [RandomizedSearchCV] {N_ITER} iteraciones x {CV_SPLITS} folds ...")
    t0 = time.time()

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

    base_model = xgb.XGBRegressor(
        tree_method  = "hist",
        random_state = RANDOM_SEED,
        eval_metric  = "rmse",
    )

    search = RandomizedSearchCV(
        estimator          = base_model,
        param_distributions= PARAM_DIST,
        n_iter             = N_ITER,
        scoring            = "neg_root_mean_squared_error",
        cv                 = tscv,
        verbose            = 0,
        random_state       = RANDOM_SEED,
        n_jobs             = -1,
        refit              = True,
    )
    search.fit(X_train, y_train)

    elapsed = time.time() - t0
    print(f"  Completado en {elapsed:.1f}s")
    print(f"  Mejor RMSE CV : {-search.best_score_:.4f} MWh")
    print(f"  Mejores params: {search.best_params_}")

    # 5.4 Re-entrenar con early stopping usando los mejores hiperparámetros
    final_model = xgb.XGBRegressor(
        **search.best_params_,
        tree_method          = "hist",
        random_state         = RANDOM_SEED,
        eval_metric          = "rmse",
        early_stopping_rounds= 50,
    )
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    trained_models[sector] = (final_model, feats, test_df, target_col)

    # 5.5 Métricas train / test
    y_pred_tr = final_model.predict(X_train)
    y_pred_te = final_model.predict(X_test)

    print("\n  [TRAIN]")
    m_tr = compute_metrics("train", y_train, y_pred_tr)
    print("\n  [TEST]")
    m_te = compute_metrics("test",  y_test,  y_pred_te)

    m_tr["sector"] = sector
    m_te["sector"] = sector
    all_metrics.extend([m_tr, m_te])

    # 5.6 Guardar modelo
    final_model.save_model(str(MODEL_DIR / f"xgboost_{sector}.json"))

# ─────────────────────────────────────────────────────────────
# 6. TABLA DE MÉTRICAS COMPLETA
# ─────────────────────────────────────────────────────────────
metrics_df = pd.DataFrame(all_metrics)[["sector", "split", "MAE", "RMSE", "R2", "MAPE", "WMAPE"]]
metrics_df.to_csv(EVAL_DIR / "metricas_todos_sectores.csv", index=False)

print("\n\n" + "=" * 65)
print("RESUMEN METRICAS TEST — todos los sectores")
print("=" * 65)
test_summary = metrics_df[metrics_df["split"] == "test"].set_index("sector")
print(test_summary.to_string())

# ─────────────────────────────────────────────────────────────
# 7. VISUALIZACIONES
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("7. GENERANDO GRAFICOS")
print("=" * 65)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "font.size": 11})

COLORS = {
    "total"      : "#2c7bb6",
    "residencial": "#1a9641",
    "servicios"  : "#d7191c",
    "industria"  : "#fdae61",
}

# 7.1 Real vs Predicho — 4 subplots apilados
fig, axes = plt.subplots(4, 1, figsize=(15, 18), sharex=False)

for ax, (sector, (model, feats, tdf, tcol)) in zip(axes, trained_models.items()):
    preds = model.predict(tdf[feats].values)
    tmp   = tdf.copy()
    tmp["pred"] = preds
    grp   = tmp.groupby("date")[[tcol, "pred"]].sum().reset_index()

    ax.plot(grp["date"], grp[tcol],   label="Real",     color=COLORS[sector], linewidth=1.5)
    ax.plot(grp["date"], grp["pred"], label="Predicho", color="black",
            linewidth=1.2, linestyle="--", alpha=0.8)
    ax.set_title(f"Sector {sector.capitalize()}", fontsize=12)
    ax.set_ylabel("MWh / dia")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

fig.suptitle("Real vs Predicho (test) — Canarias por sector", fontsize=14)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(EVAL_DIR / "01_real_vs_predicho_sectores.png", bbox_inches="tight")
plt.close()
print("   [OK] 01_real_vs_predicho_sectores.png")

# 7.2 Scatter real vs predicho — 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(13, 12))
axes = axes.flatten()

for ax, (sector, (model, feats, tdf, tcol)) in zip(axes, trained_models.items()):
    yt = tdf[tcol].values
    yp = model.predict(tdf[feats].values)
    r2 = r2_score(yt, yp)
    ax.scatter(yt, yp, alpha=0.12, s=7, color=COLORS[sector])
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    ax.plot(lims, lims, "k--", linewidth=1.3)
    ax.set_title(f"{sector.capitalize()}  (R2={r2:.4f})", fontsize=11)
    ax.set_xlabel("Real (MWh)")
    ax.set_ylabel("Predicho (MWh)")

fig.suptitle("Scatter Real vs Predicho — por sector", fontsize=13)
fig.tight_layout()
fig.savefig(EVAL_DIR / "02_scatter_sectores.png", bbox_inches="tight")
plt.close()
print("   [OK] 02_scatter_sectores.png")

# 7.3 Residuos por sector
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes = axes.flatten()

for ax, (sector, (model, feats, tdf, tcol)) in zip(axes, trained_models.items()):
    yt  = tdf[tcol].values
    yp  = model.predict(tdf[feats].values)
    res = yt - yp
    ax.hist(res, bins=50, color=COLORS[sector], edgecolor="white", alpha=0.8)
    ax.axvline(0, color="black", linestyle="--")
    ax.set_title(f"Residuos — {sector.capitalize()}", fontsize=11)
    ax.set_xlabel("Error (MWh)")
    ax.set_ylabel("Frecuencia")

fig.suptitle("Distribucion de Residuos — por sector", fontsize=13)
fig.tight_layout()
fig.savefig(EVAL_DIR / "03_residuos_sectores.png", bbox_inches="tight")
plt.close()
print("   [OK] 03_residuos_sectores.png")

# 7.4 Importancia de features — sector total (top 20)
model_total, feats_total, _, _ = trained_models["total"]
imp = (
    pd.Series(model_total.feature_importances_, index=feats_total)
    .sort_values(ascending=False)
    .head(20)
)
fig, ax = plt.subplots(figsize=(9, 7))
imp[::-1].plot(kind="barh", ax=ax, color="#2c7bb6")
ax.set_title("Top 20 features — Sector Total", fontsize=12)
ax.set_xlabel("Importancia (gain)")
fig.tight_layout()
fig.savefig(EVAL_DIR / "04_feature_importance_total.png", bbox_inches="tight")
plt.close()
print("   [OK] 04_feature_importance_total.png")

# 7.5 MAPE por municipio — sector total
_, feats_t, test_t, tcol_t = trained_models["total"]
test_t = test_t.copy()
test_t["pred"] = trained_models["total"][0].predict(test_t[feats_t].values)

mape_muni = (
    test_t.groupby("municipality")
    .apply(lambda g: mape(g[tcol_t].values, g["pred"].values), include_groups=False)
    .sort_values()
)
fig, ax = plt.subplots(figsize=(10, max(6, len(mape_muni) * 0.28)))
mape_muni.plot(kind="barh", ax=ax, color="#abd9e9", edgecolor="#2c7bb6")
ax.axvline(mape_muni.mean(), color="red", linestyle="--",
           label=f"Media: {mape_muni.mean():.1f}%")
ax.set_title("MAPE por municipio — Sector Total (test)", fontsize=12)
ax.set_xlabel("MAPE (%)")
ax.legend()
fig.tight_layout()
fig.savefig(EVAL_DIR / "05_mape_municipios_total.png", bbox_inches="tight")
plt.close()
print("   [OK] 05_mape_municipios_total.png")

# 7.6 Comparativa de métricas entre sectores
test_m      = metrics_df[metrics_df["split"] == "test"].copy()
colors_list = [COLORS[s] for s in test_m["sector"]]
fig, axes   = plt.subplots(1, 4, figsize=(17, 5))

for ax, col in zip(axes, ["MAE", "RMSE", "R2", "MAPE", "WMAPE"]):
    ax.bar(test_m["sector"], test_m[col], color=colors_list, edgecolor="white")
    ax.set_title(col, fontsize=12)
    ax.set_xlabel("Sector")
    ax.tick_params(axis="x", rotation=15)

fig.suptitle("Comparativa de metricas (test) — todos los sectores", fontsize=13)
fig.tight_layout()
fig.savefig(EVAL_DIR / "06_comparativa_metricas.png", bbox_inches="tight")
plt.close()
print("   [OK] 06_comparativa_metricas.png")

# 7.7 Guardar predicciones test de todos los sectores
preds_all = None
for sector, (model, feats, tdf, tcol) in trained_models.items():
    tmp = tdf[["municipality", "date", tcol]].copy()
    tmp[f"pred_{sector}"] = model.predict(tdf[feats].values)
    if preds_all is None:
        preds_all = tmp
    else:
        preds_all = preds_all.merge(tmp, on=["municipality", "date"], how="outer",
                                    suffixes=("", f"_{sector}"))

preds_all.to_csv(EVAL_DIR / "predicciones_test_todos.csv", index=False)

print("\n" + "=" * 65)
print("PIPELINE COMPLETADO.")
print(f"Modelos en: {MODEL_DIR}")
print(f"Evaluaciones en: {EVAL_DIR}")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# REFERENCIA: ¿Qué valores de cada métrica son buenos?
# ─────────────────────────────────────────────────────────────
print("""
+----------+---------------+---------------+-------------------+
| Metrica  |  Excelente    |  Aceptable    |  Mejorable        |
+----------+---------------+---------------+-------------------+
| MAPE     |  < 5 %        |  5 - 10 %     |  > 10 %           |
| R2       |  > 0.95       |  0.85 - 0.95  |  < 0.85           |
| MAE      |  ~5% media    |  5-10% media  |  > 10% media      |
| RMSE     |  ~8% media    |  8-15% media  |  > 15% media      |
+----------+---------------+---------------+-------------------+

NOTA: MAE y RMSE se interpretan en relacion al consumo medio del
municipio. Calcula: (MAE / mean(demand)) * 100 para normalizarlos.
Para demanda electrica municipal en Canarias, MAPE < 8% y R2 > 0.92
son resultados muy solidos para un TFC.
RMSE > MAE siempre: la diferencia indica presencia de outliers.
Si RMSE >> MAE, revisa episodios meteorologicos extremos.
""")
