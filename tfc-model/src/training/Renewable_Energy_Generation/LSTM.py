from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

TARGET_COLUMN = "ree_generacion_renovable_value"
DATE_COLUMN = "date"
DROP_COLUMNS = {DATE_COLUMN, "weather_data_source"}


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


def load_training_frame(data_path: Path, target_column: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(data_path)
    df = add_calendar_features(df)
    df = df.sort_values(DATE_COLUMN).dropna().reset_index(drop=True)

    feature_columns = [
        column
        for column in df.columns
        if column not in DROP_COLUMNS and column != target_column
    ]
    return df, feature_columns


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


def build_sequences(
    feature_matrix: np.ndarray,
    target_array: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    sequences = []
    targets = []

    for index in range(sequence_length, len(feature_matrix)):
        sequences.append(feature_matrix[index - sequence_length:index])
        targets.append(target_array[index])

    return np.asarray(sequences), np.asarray(targets)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_model(sequence_length: int, feature_count: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(sequence_length, feature_count)),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


def main() -> None:
    default_data_path, default_model_dir = resolve_paths()

    parser = argparse.ArgumentParser(
        description="Train an LSTM model for renewable energy generation forecasting."
    )
    parser.add_argument("--data-path", type=Path, default=default_data_path)
    parser.add_argument("--model-dir", type=Path, default=default_model_dir)
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    parser.add_argument("--sequence-length", type=int, default=14)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.random_state)
    tf.random.set_seed(args.random_state)

    dataframe, feature_columns = load_training_frame(args.data_path, args.target_column)
    train_df, validation_df, test_df = split_dataframe(dataframe)

    x_train_df = train_df[feature_columns]
    y_train_df = train_df[[args.target_column]]
    x_validation_df = validation_df[feature_columns]
    y_validation_df = validation_df[[args.target_column]]
    x_test_df = test_df[feature_columns]
    y_test_df = test_df[[args.target_column]]

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    x_train_scaled = feature_scaler.fit_transform(x_train_df)
    x_validation_scaled = feature_scaler.transform(x_validation_df)
    x_test_scaled = feature_scaler.transform(x_test_df)

    y_train_scaled = target_scaler.fit_transform(y_train_df)
    y_validation_scaled = target_scaler.transform(y_validation_df)
    y_test_scaled = target_scaler.transform(y_test_df)

    x_train_seq, y_train_seq = build_sequences(x_train_scaled, y_train_scaled, args.sequence_length)
    x_validation_seq, y_validation_seq = build_sequences(
        x_validation_scaled, y_validation_scaled, args.sequence_length
    )
    x_test_seq, y_test_seq = build_sequences(x_test_scaled, y_test_scaled, args.sequence_length)

    if len(x_train_seq) == 0 or len(x_validation_seq) == 0 or len(x_test_seq) == 0:
        raise ValueError("Not enough rows to build LSTM sequences. Reduce sequence length or use more data.")

    model = build_model(args.sequence_length, len(feature_columns))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        x_train_seq,
        y_train_seq,
        validation_data=(x_validation_seq, y_validation_seq),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    validation_predictions_scaled = model.predict(x_validation_seq, verbose=0)
    test_predictions_scaled = model.predict(x_test_seq, verbose=0)

    y_validation_true = target_scaler.inverse_transform(y_validation_seq)
    y_test_true = target_scaler.inverse_transform(y_test_seq)
    validation_predictions = target_scaler.inverse_transform(validation_predictions_scaled)
    test_predictions = target_scaler.inverse_transform(test_predictions_scaled)

    validation_metrics = evaluate_predictions(
        y_validation_true.flatten(),
        validation_predictions.flatten(),
    )
    test_metrics = evaluate_predictions(
        y_test_true.flatten(),
        test_predictions.flatten(),
    )

    args.model_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model_dir / "lstm_renewable_generation.keras"
    feature_scaler_path = args.model_dir / "lstm_renewable_generation_feature_scaler.joblib"
    target_scaler_path = args.model_dir / "lstm_renewable_generation_target_scaler.joblib"
    metadata_path = args.model_dir / "lstm_renewable_generation_metadata.json"

    model.save(model_path)
    joblib.dump(feature_scaler, feature_scaler_path)
    joblib.dump(target_scaler, target_scaler_path)

    metadata = {
        "model_type": "LSTM",
        "target_column": args.target_column,
        "feature_columns": feature_columns,
        "sequence_length": args.sequence_length,
        "dataset_path": str(args.data_path),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "history": {
            "loss": [float(value) for value in history.history.get("loss", [])],
            "val_loss": [float(value) for value in history.history.get("val_loss", [])],
            "mae": [float(value) for value in history.history.get("mae", [])],
            "val_mae": [float(value) for value in history.history.get("val_mae", [])],
        },
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("LSTM training completed.")
    print(f"Model saved to: {model_path}")
    print(f"Feature scaler saved to: {feature_scaler_path}")
    print(f"Target scaler saved to: {target_scaler_path}")
    print(f"Metadata saved to: {metadata_path}")
    print("Validation metrics:")
    print(json.dumps(validation_metrics, indent=2))
    print("Test metrics:")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
