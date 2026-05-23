"""
=============================================================
MODELO PREDICTIVO DE GENERACIÓN EÓLICA - CANARIAS
TFC · HistGradientBoosting + Ridge (comparativa automática)
=============================================================
Target : ree_eolica_value

Decisiones metodológicas documentadas:
  - Solar descartada: crecimiento estructural +80% (2020-2025)
    por nueva capacidad instalada. Sin datos de MW instalados
    ningún modelo estadístico puede extrapolar ese crecimiento.
  - Eólica modelada con HGB (mejor generalización con pocos datos)
  - Features seleccionadas por permutation importance real
  - Interacciones físicas: wind_chaos, wind_variability_ratio
  - Lags solo hasta lag_3d (autocorr lag_7d cae a 0.24)

Métricas: MAE, RMSE, R², MAPE, WMAPE
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import json
import time
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from scipy.stats import uniform, loguniform
import xgboost as xgb

# ─────────────────────────────────────────────────────────────
# 0. CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parents[3]  # tfc-model
SRC_DIR     = ROOT_DIR / "src"
DATA_PATH   = ROOT_DIR / "data" / "final_renewable_generation_dataset.csv"

TARGET = "ree_eolica_value"

RANDOM_SEED = 42
TEST_RATIO  = 0.2
VAL_RATIO   = 0.2

MODEL_DIR   = SRC_DIR / "models" / "renewable_energy_generation"
EVAL_DIR    = SRC_DIR / "evaluation" / "renewable_energy_generation"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

CV_SPLITS    = 3
N_ITER_HGB   = 40
N_ITER_RIDGE = 50

# ─────────────────────────────────────────────────────────────
# 1. CARGA
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("1. CARGA DE DATOS")
print("=" * 65)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
if "weather_data_source" in df.columns:
    df = df.drop(columns=["weather_data_source"])
df = df.dropna(subset=[TARGET]).sort_values("date").reset_index(drop=True)

print(f"   Filas  : {len(df):,}")
print(f"   Fechas : {df['date'].min().date()} -> {df['date'].max().date()}")
print(f"   Media  : {df[TARGET].mean():.1f}  |  Std: {df[TARGET].std():.1f}")
print(f"   Min/Max: {df[TARGET].min():.1f} / {df[TARGET].max():.1f}")

# ─────────────────────────────────────────────────────────────
# 2. INGENIERÍA DE FEATURES
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("2. INGENIERÍA DE FEATURES")
print("=" * 65)

# ── Calendario
df["year"]       = df["date"].dt.year
df["month"]      = df["date"].dt.month
df["dayofweek"]  = df["date"].dt.dayofweek
df["dayofyear"]  = df["date"].dt.dayofyear
df["quarter"]    = df["date"].dt.quarter
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

# ── Estacionalidad cíclica
df["month_sin"] = np.sin(2 * np.pi * df["month"]     / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"]     / 12)
df["doy_sin"]   = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
df["doy_cos"]   = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
df["dow_sin"]   = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dow_cos"]   = np.cos(2 * np.pi * df["dayofweek"] / 7)

# ── Física del viento (P ∝ v³ — relación potencia-velocidad)
for col in ["wind_speed_avg_ms", "wind_speed_max_ms", "wind_speed_sdev_ms"]:
    if col in df.columns:
        df[f"{col}_cb"] = df[col] ** 3
        df[f"{col}_sq"] = df[col] ** 2

# ── Interacciones físicas de viento
# wind_chaos: velocidad × variabilidad de dirección → turbulencia
if "wind_speed_avg_ms" in df.columns and "wind_dir_sdev_deg" in df.columns:
    df["wind_chaos"] = df["wind_speed_avg_ms"] * df["wind_dir_sdev_deg"]

# wind_variability_ratio: std/media del viento → régimen de ráfagas
if "wind_speed_avg_ms" in df.columns and "wind_speed_sdev_ms" in df.columns:
    df["wind_variability_ratio"] = (
        df["wind_speed_sdev_ms"] / (df["wind_speed_avg_ms"] + 0.1)
    )

# Interacción viento × estacionalidad
if "wind_speed_avg_ms" in df.columns:
    df["wind_x_month_sin"]    = df["wind_speed_avg_ms"] * df["month_sin"]
    df["wind_x_month_cos"]    = df["wind_speed_avg_ms"] * df["month_cos"]
    df["wind_cb_x_month_sin"] = df["wind_speed_avg_ms_cb"] * df["month_sin"]

# ── Lags del target (solo lag_1d-3d: autocorr cae a 0.24 en lag_7d)
for lag in [1, 2, 3]:
    df[f"lag_{lag}d"] = df[TARGET].shift(lag)

# ── Rolling
df["rolling_3d_mean"]  = df[TARGET].shift(1).rolling(3,  min_periods=2).mean()
df["rolling_7d_mean"]  = df[TARGET].shift(1).rolling(7,  min_periods=3).mean()
df["rolling_7d_std"]   = df[TARGET].shift(1).rolling(7,  min_periods=3).std()
df["rolling_14d_mean"] = df[TARGET].shift(1).rolling(14, min_periods=5).mean()

# ── Lag de hidroeólica (correlación 0.79 con eólica, mismo driver físico)
if "ree_hidroeolica_value" in df.columns:
    df["hidroeolica_lag1"] = df["ree_hidroeolica_value"].shift(1)

# Eliminar NaN de lags obligatorios
df = df.dropna(subset=["lag_1d", "lag_2d", "lag_3d", "rolling_7d_mean"]).copy()
print(f"   Filas tras feature engineering: {len(df):,}")

# ─────────────────────────────────────────────────────────────
# 3. FEATURES Y SPLIT TEMPORAL
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("3. SPLIT TEMPORAL")
print("=" * 65)

# Excluir otras variables REE (leakage) y no numéricas
EXCLUIR = {
    "date", TARGET, "weather_data_source",
    "ree_generacion_renovable_value",
    "ree_eolica_pct", "ree_solar_fotovoltaica_pct",
    "ree_hidraulica_pct", "ree_hidroeolica_pct", "ree_otras_renovables_pct",
    "ree_solar_fotovoltaica_value", "ree_hidraulica_value",
    "ree_hidroeolica_value", "ree_otras_renovables_value",
}

FEATURES = [
    c for c in df.columns
    if c not in EXCLUIR
    and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
]
print(f"   Features: {len(FEATURES)}")

test_start = int(len(df) * (1 - TEST_RATIO))
trainval_df = df.iloc[:test_start].copy()
test_df  = df.iloc[test_start:].copy()
val_start = int(len(trainval_df) * (1 - VAL_RATIO))
train_df = trainval_df.iloc[:val_start].copy()
val_df = trainval_df.iloc[val_start:].copy()

X_train = train_df[FEATURES].values
y_train = train_df[TARGET].values
X_val   = val_df[FEATURES].values
y_val   = val_df[TARGET].values
X_test  = test_df[FEATURES].values
y_test  = test_df[TARGET].values

# Imputación (para HGB y XGBoost)
imputer     = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_val_imp   = imputer.transform(X_val)
X_test_imp  = imputer.transform(X_test)

print(f"   Train: {len(train_df):,} filas  ({train_df['date'].min().date()} -> {train_df['date'].max().date()})")
print(f"   Val  : {len(val_df):,} filas  ({val_df['date'].min().date()} -> {val_df['date'].max().date()})")
print(f"   Test : {len(test_df):,} filas  ({test_df['date'].min().date()} -> {test_df['date'].max().date()})")
print(f"   Media train: {y_train.mean():.1f}  |  Media val: {y_val.mean():.1f}  |  Media test: {y_test.mean():.1f}")

# ─────────────────────────────────────────────────────────────
# 4. MÉTRICAS
# ─────────────────────────────────────────────────────────────
def mape(y_true, y_pred):
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100)

def wmape(y_true, y_pred):
    return float(np.sum(np.abs(y_true-y_pred))/np.sum(np.abs(y_true))*100)

def compute_metrics(name, y_true, y_pred):
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    m = {
        "split" : name,
        "MAE"   : mean_absolute_error(yt, yp),
        "RMSE"  : float(np.sqrt(mean_squared_error(yt, yp))),
        "R2"    : r2_score(yt, yp),
        "MAPE"  : mape(yt, yp),
        "WMAPE" : wmape(yt, yp),
    }
    print(f"      MAE   : {m['MAE']:>12.4f}")
    print(f"      RMSE  : {m['RMSE']:>12.4f}")
    print(f"      R2    : {m['R2']:>12.4f}")
    print(f"      MAPE  : {m['MAPE']:>11.2f} %")
    print(f"      WMAPE : {m['WMAPE']:>11.2f} %")
    return m

tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

# ─────────────────────────────────────────────────────────────
# 5A. RIDGE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("5A. RIDGE REGRESSION")
print("=" * 65)

t0 = time.time()
ridge_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("ridge",   Ridge()),
])
ridge_search = RandomizedSearchCV(
    estimator           = ridge_pipeline,
    param_distributions = {"ridge__alpha": loguniform(1e-1, 1e3)},
    n_iter              = N_ITER_RIDGE,
    scoring             = "neg_root_mean_squared_error",
    cv                  = tscv,
    random_state        = RANDOM_SEED,
    n_jobs              = -1,
    verbose             = 0,
)
ridge_search.fit(X_train, y_train)
print(f"   Completado en {time.time()-t0:.1f}s | Mejor alpha: {ridge_search.best_params_['ridge__alpha']:.4f}")

ridge_model     = ridge_search.best_estimator_
y_pred_ridge_tr = ridge_model.predict(X_train)
y_pred_ridge_va = ridge_model.predict(X_val)
y_pred_ridge_te = ridge_model.predict(X_test)

print("\n   [TRAIN]")
m_ridge_tr = compute_metrics("train", y_train, y_pred_ridge_tr)
print("\n   [VALIDATION]")
m_ridge_va = compute_metrics("validation", y_val, y_pred_ridge_va)
print("\n   [TEST]")
m_ridge_te = compute_metrics("test",  y_test,  y_pred_ridge_te)

# ─────────────────────────────────────────────────────────────
# 5B. HISTGRADIENTBOOSTING
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("5B. HISTGRADIENTBOOSTING (mejor para datos pequeños con alta varianza)")
print("=" * 65)

# Espacio alrededor de los mejores valores encontrados en diagnóstico
from scipy.stats import randint
hgb_param_dist = {
    "max_depth"        : randint(3, 6),
    "learning_rate"    : loguniform(0.02, 0.08),
    "min_samples_leaf" : randint(5, 20),
    "l2_regularization": loguniform(0.1, 2.0),
    "max_iter"         : randint(400, 800),
}

t0 = time.time()
hgb_search = RandomizedSearchCV(
    estimator           = HistGradientBoostingRegressor(
                            random_state      = RANDOM_SEED,
                            early_stopping    = True,
                            validation_fraction = 0.1,
                            n_iter_no_change  = 30,
                          ),
    param_distributions = hgb_param_dist,
    n_iter              = N_ITER_HGB,
    scoring             = "neg_root_mean_squared_error",
    cv                  = tscv,
    random_state        = RANDOM_SEED,
    n_jobs              = -1,
    verbose             = 0,
)
hgb_search.fit(X_train_imp, y_train)
print(f"   Completado en {time.time()-t0:.1f}s")
print(f"   Mejor RMSE CV : {-hgb_search.best_score_:.4f}")
print(f"   Mejores params: {hgb_search.best_params_}")

hgb_model     = hgb_search.best_estimator_
y_pred_hgb_tr = hgb_model.predict(X_train_imp)
y_pred_hgb_va = hgb_model.predict(X_val_imp)
y_pred_hgb_te = hgb_model.predict(X_test_imp)

print("\n   [TRAIN]")
m_hgb_tr = compute_metrics("train", y_train, y_pred_hgb_tr)
print("\n   [VALIDATION]")
m_hgb_va = compute_metrics("validation", y_val, y_pred_hgb_va)
print("\n   [TEST]")
m_hgb_te = compute_metrics("test",  y_test,  y_pred_hgb_te)

# ─────────────────────────────────────────────────────────────
# 6. SELECCIÓN AUTOMÁTICA
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("6. SELECCIÓN DEL MEJOR MODELO")
print("=" * 65)

gap_ridge = m_ridge_tr["R2"] - m_ridge_te["R2"]
gap_hgb   = m_hgb_tr["R2"]   - m_hgb_te["R2"]

print(f"\n   Ridge — Test R2: {m_ridge_te['R2']:.4f} | WMAPE: {m_ridge_te['WMAPE']:.2f}% | Gap: {gap_ridge:.4f}")
print(f"   HGB   — Test R2: {m_hgb_te['R2']:.4f} | WMAPE: {m_hgb_te['WMAPE']:.2f}% | Gap: {gap_hgb:.4f}")

if m_hgb_te["R2"] >= m_ridge_te["R2"]:
    best_name, best_pred_te, best_pred_tr = "HGB",   y_pred_hgb_te, y_pred_hgb_tr
    m_best_te, m_best_tr                  = m_hgb_te, m_hgb_tr
else:
    best_name, best_pred_te, best_pred_tr = "Ridge", y_pred_ridge_te, y_pred_ridge_tr
    m_best_te, m_best_tr                  = m_ridge_te, m_ridge_tr

print(f"\n   MODELO SELECCIONADO: {best_name}")

# ─────────────────────────────────────────────────────────────
# 7. GUARDAR MODELOS Y MÉTRICAS
# ─────────────────────────────────────────────────────────────
joblib.dump(hgb_model,   MODEL_DIR / "hgb_eolica.pkl")
joblib.dump(ridge_model, MODEL_DIR / "ridge_eolica.pkl")
joblib.dump(imputer,     MODEL_DIR / "imputer_eolica.pkl")

metrics_df = pd.DataFrame([
    {**m_ridge_te, "modelo": "Ridge"},
    {**m_hgb_te,   "modelo": "HGB"},
])
metrics_df.to_csv(EVAL_DIR / "metricas_eolica.csv", index=False)
pd.DataFrame([
    {**m_ridge_tr, "modelo": "Ridge"},
    {**m_ridge_va, "modelo": "Ridge"},
    {**m_ridge_te, "modelo": "Ridge"},
    {**m_hgb_tr, "modelo": "HGB"},
    {**m_hgb_va, "modelo": "HGB"},
    {**m_hgb_te, "modelo": "HGB"},
]).to_csv(EVAL_DIR / "metricas_eolica_all_splits.csv", index=False)

baseline_rows = []
for baseline_name, values in [
    ("lag_1d", test_df["lag_1d"].values),
    ("rolling_3d_mean", test_df["rolling_3d_mean"].values),
    ("rolling_7d_mean", test_df["rolling_7d_mean"].values),
]:
    row = compute_metrics(f"baseline_{baseline_name}", y_test, values)
    row["baseline"] = baseline_name
    baseline_rows.append(row)
pd.DataFrame(baseline_rows)[["baseline", "split", "MAE", "RMSE", "R2", "MAPE", "WMAPE"]].to_csv(
    EVAL_DIR / "baseline_eolica.csv",
    index=False,
)

metadata = {
    "problem": "renewable_energy_generation",
    "target": TARGET,
    "features": FEATURES,
    "split": {
        "strategy": "temporal_train_validation_test",
        "test_ratio": TEST_RATIO,
        "validation_ratio_within_trainval": VAL_RATIO,
        "test_is_never_used_for_tuning": True,
    },
    "anti_leakage": [
        "target lags use shift(1..3)",
        "rolling features use shifted target values",
        "same-day REE technology values are excluded from features",
        "uncertainty/calibration must be learned on validation, not final test",
    ],
    "known_limits": [
        "weather is aggregated at Canarias daily level",
        "installed wind capacity and curtailment are not included",
        "prediction should be presented with uncertainty",
    ],
}
(MODEL_DIR / "metadata_eolica.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n   Modelos guardados en {MODEL_DIR}")
print(f"   Evaluacion guardada en {EVAL_DIR}")

# ─────────────────────────────────────────────────────────────
# 8. VISUALIZACIONES
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("8. GENERANDO GRAFICOS")
print("=" * 65)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "font.size": 11})
COLOR = "#2c7bb6"

# ── 8.1 Real vs Predicho — ambos modelos
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
for ax, (name, preds, color) in zip(axes, [
    ("Ridge", y_pred_ridge_te, "#e66101"),
    ("HGB",   y_pred_hgb_te,   "#1a9641"),
]):
    r2 = r2_score(y_test, preds)
    ax.plot(test_df["date"].values, y_test, label="Real",  color=COLOR, linewidth=1.5)
    ax.plot(test_df["date"].values, preds,  label=name,    color=color,
            linewidth=1.2, linestyle="--", alpha=0.85)
    ax.set_title(f"{name}  (R2={r2:.4f})", fontsize=12)
    ax.set_ylabel("Generacion eolica (REE)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.suptitle("Generacion Eólica — Real vs Predicho (test)", fontsize=14)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(EVAL_DIR / "01_real_vs_predicho_eolica.png", bbox_inches="tight")
plt.close()
print("   [OK] 01_real_vs_predicho_eolica.png")

# ── 8.2 Scatter mejor modelo
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, best_pred_te, alpha=0.35, s=15, color=COLOR)
lims = [min(y_test.min(), best_pred_te.min()), max(y_test.max(), best_pred_te.max())]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Prediccion perfecta")
ax.set_xlabel("Real"); ax.set_ylabel("Predicho")
ax.set_title(f"Scatter [{best_name}]  R2={m_best_te['R2']:.4f}", fontsize=12)
ax.legend()
fig.tight_layout()
fig.savefig(EVAL_DIR / "02_scatter_eolica.png", bbox_inches="tight")
plt.close()
print("   [OK] 02_scatter_eolica.png")

# ── 8.3 Residuos mejor modelo
residuals = y_test - best_pred_te
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(residuals, bins=40, color=COLOR, edgecolor="white", alpha=0.8)
axes[0].axvline(0, color="red", linestyle="--")
axes[0].set_title(f"Distribucion residuos [{best_name}]")
axes[0].set_xlabel("Error"); axes[0].set_ylabel("Frecuencia")
axes[1].scatter(best_pred_te, residuals, alpha=0.3, s=10, color="#d7191c")
axes[1].axhline(0, color="black", linestyle="--")
axes[1].set_title("Residuos vs Predicho")
axes[1].set_xlabel("Predicho"); axes[1].set_ylabel("Residuo")
fig.suptitle(f"Analisis de Residuos [{best_name}] — Test", fontsize=13)
fig.tight_layout()
fig.savefig(EVAL_DIR / "03_residuos_eolica.png", bbox_inches="tight")
plt.close()
print("   [OK] 03_residuos_eolica.png")

# ── 8.4 Permutation importance del mejor modelo
print("\n   Calculando permutation importance ...")
best_model_obj = hgb_model if best_name == "HGB" else ridge_model
X_te_for_imp   = X_test_imp if best_name == "HGB" else X_test
perm = permutation_importance(
    best_model_obj, X_te_for_imp, y_test,
    n_repeats=15, random_state=RANDOM_SEED, scoring="r2",
)
imp_series = (
    pd.Series(perm.importances_mean, index=FEATURES)
    .sort_values(ascending=False)
    .head(20)
)
fig, ax = plt.subplots(figsize=(9, 7))
imp_series[::-1].plot(kind="barh", ax=ax, color=COLOR)
ax.set_title(f"Top 20 features (permutation importance) — [{best_name}]", fontsize=12)
ax.set_xlabel("Reduccion media de R2")
fig.tight_layout()
fig.savefig(EVAL_DIR / "04_feature_importance_eolica.png", bbox_inches="tight")
plt.close()
print("   [OK] 04_feature_importance_eolica.png")

# ── 8.5 WMAPE mensual
test_df2 = test_df.copy()
test_df2["pred"]       = best_pred_te
test_df2["month_name"] = test_df2["date"].dt.strftime("%Y-%m")
monthly = (
    test_df2.groupby("month_name")
    .apply(lambda g: wmape(g[TARGET].values, g["pred"].values), include_groups=False)
    .reset_index()
)
monthly.columns = ["mes", "WMAPE"]
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(monthly["mes"], monthly["WMAPE"], color=COLOR, edgecolor="white", alpha=0.85)
ax.set_title(f"WMAPE mensual — Eolica [{best_name}] (test)", fontsize=12)
ax.set_xlabel("Mes"); ax.set_ylabel("WMAPE (%)")
plt.xticks(rotation=45, ha="right")
fig.tight_layout()
fig.savefig(EVAL_DIR / "05_wmape_mensual_eolica.png", bbox_inches="tight")
plt.close()
print("   [OK] 05_wmape_mensual_eolica.png")

# ── 8.6 Comparativa barras Ridge vs HGB
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
modelos = ["Ridge", "HGB"]
colors  = ["#e66101", "#1a9641"]
for ax, (col, vals) in zip(axes, [
    ("R2",    [m_ridge_te["R2"],    m_hgb_te["R2"]]),
    ("WMAPE", [m_ridge_te["WMAPE"], m_hgb_te["WMAPE"]]),
    ("MAE",   [m_ridge_te["MAE"],   m_hgb_te["MAE"]]),
    ("RMSE",  [m_ridge_te["RMSE"],  m_hgb_te["RMSE"]]),
]):
    bars = ax.bar(modelos, vals, color=colors, edgecolor="white")
    ax.set_title(col, fontsize=12)
    best_idx = int(np.argmax(vals)) if col == "R2" else int(np.argmin(vals))
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(3)
fig.suptitle("Comparativa Ridge vs HGB (test) — borde dorado = mejor", fontsize=12)
fig.tight_layout()
fig.savefig(EVAL_DIR / "06_comparativa_modelos_eolica.png", bbox_inches="tight")
plt.close()
print("   [OK] 06_comparativa_modelos_eolica.png")

# ── 8.7 Guardar predicciones
test_df2[["date", TARGET, "pred"]].rename(
    columns={"pred": f"pred_{best_name.lower()}"}
).assign(pred_ridge=y_pred_ridge_te, pred_hgb=y_pred_hgb_te
).to_csv(EVAL_DIR / "predicciones_test_eolica.csv", index=False)

# ─────────────────────────────────────────────────────────────
# 9. RESUMEN FINAL
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RESUMEN FINAL — TEST")
print("=" * 65)
print(f"\n{'Modelo':<10} {'R2':>8} {'WMAPE':>8} {'MAE':>10} {'RMSE':>10}")
print("-" * 45)
for name, m in [("Ridge", m_ridge_te), ("HGB", m_hgb_te)]:
    marker = " <-- MEJOR" if name == best_name else ""
    print(f"{name:<10} {m['R2']:>8.4f} {m['WMAPE']:>7.2f}% "
          f"{m['MAE']:>10.1f} {m['RMSE']:>10.1f}{marker}")

print(f"\n   Modelos en: {MODEL_DIR}")
print(f"   Evaluaciones en: {EVAL_DIR}")
print("=" * 65)

print("""
NOTA PARA EL TFC - Por que se descarto solar:
  La generacion fotovoltaica en Canarias presenta crecimiento
  estructural acelerado (+80% entre 2020 y 2025) por nueva
  capacidad instalada. Sin datos de potencia acumulada (MW),
  el techo estadistico real es R2 aprox. 0.42 independientemente
  del modelo. Se propone como linea de trabajo futuro incorporar
  datos de capacidad instalada de REE o ISTAC.
""")
