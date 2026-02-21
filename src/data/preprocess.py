import pandas as pd
import numpy as np
from pathlib import Path

# ── Column definitions ───────────────────────────────────────────────
COLUMNS = (
    ["engine_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Sensors with near-zero variance across FD001 — drop them
DROP_SENSORS = ["sensor_1", "sensor_5", "sensor_6", "sensor_10",
                "sensor_16", "sensor_18", "sensor_19"]


def load_raw(filepath: str) -> pd.DataFrame:
    """Load a raw CMAPSS text file into a DataFrame."""
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMNS)
    return df


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Remaining Useful Life for each row.
    RUL = (max cycle for that engine) - (current cycle)
    """
    max_cycles = df.groupby("engine_id")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycles, on="engine_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def drop_low_variance_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Remove sensors that carry no useful signal in FD001."""
    return df.drop(columns=DROP_SENSORS, errors="ignore")


def normalize_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Min-max normalize all remaining sensor columns per engine.
    We normalize globally here (fit on train, apply to test separately).
    """
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    df[sensor_cols] = (df[sensor_cols] - df[sensor_cols].min()) / (
        df[sensor_cols].max() - df[sensor_cols].min() + 1e-8
    )
    return df


def preprocess(input_path: str, output_path: str, is_train: bool = True):
    """Full preprocessing pipeline."""
    print(f"Loading data from {input_path}...")
    df = load_raw(input_path)

    if is_train:
        df = add_rul(df)

    df = drop_low_variance_sensors(df)
    df = normalize_sensors(df)

    # Save versioned output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path} — shape: {df.shape}")
    return df


if __name__ == "__main__":
    preprocess(
        input_path="data/raw/train_FD001.txt",
        output_path="data/processed/train_FD001_v1.csv",
        is_train=True,
    )
    preprocess(
        input_path="data/raw/test_FD001.txt",
        output_path="data/processed/test_FD001_v1.csv",
        is_train=False,
    )