import pandas as pd
from src.data.preprocess import add_rul, drop_low_variance_sensors

def test_rul_calculation():
    """RUL should be 0 at the last cycle of each engine."""
    df = pd.DataFrame({
        "engine_id": [1, 1, 1, 2, 2],
        "cycle":     [1, 2, 3, 1, 2],
    })
    result = add_rul(df)
    last_cycles = result[result["cycle"] == result.groupby("engine_id")["cycle"].transform("max")]
    assert (last_cycles["RUL"] == 0).all()

def test_drop_sensors():
    """Dropped sensors should not appear in output."""
    df = pd.DataFrame({"sensor_1": [1], "sensor_2": [2], "sensor_5": [3]})
    result = drop_low_variance_sensors(df)
    assert "sensor_1" not in result.columns
    assert "sensor_2" in result.columns