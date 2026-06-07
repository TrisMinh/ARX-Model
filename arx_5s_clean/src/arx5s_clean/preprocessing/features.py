from __future__ import annotations

import numpy as np
import pandas as pd


def add_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    timestamp = pd.to_datetime(df["Timestamp"])
    hour = timestamp.dt.hour + timestamp.dt.minute / 60.0 + timestamp.dt.second / 3600.0
    day_num = df["Day_Index"].astype(float)

    df["Light_log"] = np.log1p(df["Light_In"].clip(lower=0))
    df["TempIn_x_HumiIn"] = df["Temperature_In"] * df["Humidity_In"]
    df["TempIn_x_Light"] = df["Temperature_In"] * df["Light_log"]
    df["HumiIn_x_Light"] = df["Humidity_In"] * df["Light_log"]
    df["Indoor_Dryness"] = 100.0 - df["Humidity_In"]
    df["VPD_Proxy_In"] = (df["Temperature_In"] - 20.0).clip(lower=0.0) * df["Indoor_Dryness"].clip(lower=0.0) / 100.0
    df["Hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["Hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    df["Day_sin"] = np.sin(2.0 * np.pi * day_num / 7.0)
    df["Day_cos"] = np.cos(2.0 * np.pi * day_num / 7.0)
    return df

