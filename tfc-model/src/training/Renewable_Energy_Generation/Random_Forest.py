from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET_COLUMN = "ree_generacion_renovable_value"
DATE_COLUMN = "date"
DROP_COLUMNS = {DATE_COLUMN, "weather_data_source"}
DEFAULT_LAGS = (1, 2, 3, 7, 14, 21, 28)


def resolve_paths() -> tuple[Path, Path]:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[3]
    data_path = project_root / "data" / "final_renewable_generation_dataset.csv"
    model_dir = project_root / "src" / "models" / "renewable_energy_generation"
    return data_path, model_dir


def add_calendar_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    day_of_week = df[DATE_COLUMN].dt.dayofweek
    day_of_year = df[DATE_COLUMN].dt.dayofyear

    df["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    df["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return df


def add_lag_features(dataframe: pd.DataFrame, target_column: str, lags: tuple[int, ...]) -> pd.DataFrame:
    df = dataframe.copy()
    for lag in lags:
        df[f"{target_column}_lag_{lag}"] = df[target_column].shift(lag)
    return df


def load_training_frame(data_path: Path, target_column: str, lags: tuple[int, ...]) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df = add_calendar_features(df)
    df = add_lag_features(df, target_column, lags)
    df = df.sort_values(DATE_COLUMN).dropna().reset_index(drop=True)
    return df


def split_dataframe(
    dataframe: pd.DataFrame,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = int(len(dataframe) * train_ratio)
    validation_end = int(len(dataframe) * (train_ratio + validation_ratio))

    train_df = dataframe.iloc[:train_end].copy()
    validation_df = dataframe.iloc[train_end:validation_end].copy()
    test_df = dataframe.iloc[validation_end:].copy()
    return train_df, validation_df, test_df


def build_feature_matrices(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, list[str]]:
    feature_columns = [
        column
        for column in train_df.columns
        if column not in DROP_COLUMNS and column != target_column
    ]

    x_train = train_df[feature_columns]
    x_validation = validation_df[feature_columns]
    x_test = test_df[feature_columns]

    y_train = train_df[target_column]
    y_validation = validation_df[target_column]
    y_test = test_df[target_column]
    return x_train, x_validation, x_test, y_train, y_validation, y_test, feature_columns


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    default_data_path, default_model_dir = resolve_paths()

    parser = argparse.ArgumentParser(
        description="Train a Random Forest model for renewable energy generation forecasting."
    )
    parser.add_argument("--data-path", type=Path, default=default_data_path)
    parser.add_argument("--model-dir", type=Path, default=default_model_dir)
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=18)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    dataframe = load_training_frame(args.data_path, args.target_column, DEFAULT_LAGS)
    train_df, validation_df, test_df = split_dataframe(dataframe)

    x_train, x_validation, x_test, y_train, y_validation, y_test, feature_columns = build_feature_matrices(
        train_df,
        validation_df,
        test_df,
        args.target_column,
    )

    x_train_full = pd.concat([x_train, x_validation], axis=0)
    y_train_full = pd.concat([y_train, y_validation], axis=0)

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(x_train_full, y_train_full)

    validation_predictions = model.predict(x_validation)
    test_predictions = model.predict(x_test)

    validation_metrics = evaluate_predictions(y_validation, validation_predictions)
    test_metrics = evaluate_predictions(y_test, test_predictions)

    args.model_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model_dir / "random_forest_renewable_generation.joblib"
    metadata_path = args.model_dir / "random_forest_renewable_generation_metadata.json"

    joblib.dump(model, model_path)

    metadata = {
        "model_type": "RandomForestRegressor",
        "target_column": args.target_column,
        "feature_columns": feature_columns,
        "lag_days": list(DEFAULT_LAGS),
        "dataset_path": str(args.data_path),
        "train_rows": int(len(x_train)),
        "validation_rows": int(len(x_validation)),
        "test_rows": int(len(x_test)),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Random Forest training completed.")
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")
    print("Validation metrics:")
    print(json.dumps(validation_metrics, indent=2))
    print("Test metrics:")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
