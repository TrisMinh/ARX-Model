from __future__ import annotations

import numpy as np
import pandas as pd


# Tạo feature đầu vào cho model ARX.
def add_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    timestamp = pd.to_datetime(df["Timestamp"])
    hour = timestamp.dt.hour + timestamp.dt.minute / 60.0 + timestamp.dt.second / 3600.0
    if "Day_Index" in df.columns:
        day_num = df["Day_Index"].astype(float)
    else:
        elapsed_seconds = (timestamp - timestamp.iloc[0]).dt.total_seconds()
        day_num = np.floor(elapsed_seconds / 86400.0)

    df["Light_log"] = np.log1p(df["Light"].clip(lower=0))
    df["Temp_x_Humidity"] = df["Temperature"] * df["Humidity"]
    df["Temp_x_Light"] = df["Temperature"] * df["Light_log"]
    df["Humidity_x_Light"] = df["Humidity"] * df["Light_log"]
    df["Air_Dryness"] = 100.0 - df["Humidity"]
    df["Temp_x_Air_Dryness"] = (df["Temperature"] - 20.0).clip(lower=0.0) * df["Air_Dryness"].clip(lower=0.0) / 100.0
    df["Hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["Hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    df["Day_sin"] = np.sin(2.0 * np.pi * day_num / 7.0)
    df["Day_cos"] = np.cos(2.0 * np.pi * day_num / 7.0)
    return df
