import pandas as pd
import numpy as np
from src.training.train import build_features, evaluate

def make_dummy_df():
    """Create a minimal dataframe mimicking processed CMAPSS data."""
    rows = []
    for engine_id in range(1, 4):
        for cycle in range(1, 11):
            row = {"engine_id": engine_id, "cycle": cycle}
            for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]:
                row[f"sensor_{i}"] = np.random.rand()
            for i in range(1, 4):
                row[f"op_setting_{i}"] = np.random.rand()
            row["RUL"] = 10 - cycle
            rows.append(row)
    return pd.DataFrame(rows)


def test_build_features_shape():
    """Feature matrix should have more columns than raw sensors due to rolling features."""
    df = make_dummy_df()
    X, y, feature_cols = build_features(df)
    assert X.shape[0] == len(df)
    assert len(feature_cols) > 14  # more than raw sensors alone


def test_evaluate_metrics():
    """Perfect predictions should give RMSE=0 and R2=1."""
    y = np.array([10, 20, 30, 40])
    metrics = evaluate(y, y)
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0