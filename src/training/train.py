import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from pathlib import Path
import json

# ── Config ────────────────────────────────────────────────────────────
DATA_PATH = "data/processed/train_FD001_v1.csv"
EXPERIMENT_NAME = "predictive-maintenance-fd001"

PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42,
}

# ── Feature Engineering ───────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create rolling window features from sensor readings.
    MLflow will log which features were used.
    """
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    op_cols = [c for c in df.columns if c.startswith("op_setting_")]

    # Rolling stats over last 5 cycles per engine
    for col in sensor_cols:
        df[f"{col}_mean5"] = (
            df.groupby("engine_id")[col]
            .transform(lambda x: x.rolling(5, min_periods=1).mean())
        )
        df[f"{col}_std5"] = (
            df.groupby("engine_id")[col]
            .transform(lambda x: x.rolling(5, min_periods=1).std().fillna(0))
        )

    feature_cols = sensor_cols + op_cols + \
                   [c for c in df.columns if "_mean5" in c or "_std5" in c]

    X = df[feature_cols]
    y = df["RUL"]
    return X, y, feature_cols


# ── Evaluation ────────────────────────────────────────────────────────
def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ── Main Training Run ─────────────────────────────────────────────────
def train():
    print("Loading processed data...")
    df = pd.read_csv(DATA_PATH)

    X, y, feature_cols = build_features(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Set MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="xgboost-baseline"):

        print("Training XGBoost model...")
        model = xgb.XGBRegressor(**PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        # ── Log params ──────────────────────────────────────────
        mlflow.log_params(PARAMS)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("num_features", len(feature_cols))

        # ── Log metrics ─────────────────────────────────────────
        train_metrics = evaluate(y_train, model.predict(X_train))
        val_metrics = evaluate(y_val, model.predict(X_val))

        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        print(f"\nTrain RMSE: {train_metrics['rmse']:.2f}")
        print(f"Val   RMSE: {val_metrics['rmse']:.2f}")
        print(f"Val   MAE : {val_metrics['mae']:.2f}")
        print(f"Val   R2  : {val_metrics['r2']:.3f}")

        # ── Log feature list as artifact ─────────────────────────
        Path("artifacts").mkdir(exist_ok=True)
        with open("artifacts/features.json", "w") as f:
            json.dump(feature_cols, f)
        mlflow.log_artifact("artifacts/features.json")

        # ── Register model ───────────────────────────────────────
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name="rul-predictor",
        )

        print("\n✅ Run logged to MLflow successfully")
        print(f"   Experiment : {EXPERIMENT_NAME}")
        print(f"   Model      : rul-predictor (registered)")


if __name__ == "__main__":
    train()