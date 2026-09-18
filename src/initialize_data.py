import datetime as dt

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


def create_wind_speed_data(
    years=20,
    delay_h=16,
    delay_m=4,
    delay_y=3,
    period_y=7,
    amplitude_h=1,
    amplitude_m=0.75,
    amplitude_y=0.5,
    mean_ws=6,
    std_noise=0.5,
):
    # Define the start and end dates and create an empty dataframe with the correct hourly index
    ts_start = "2000-01-01 00:00:00"
    ts_end = (
        dt.datetime.strptime(ts_start, "%Y-%m-%d %H:%M:%S")
        + relativedelta(months=years * 12)
        - dt.timedelta(hours=1)
    ).strftime("%Y-%m-%d %H:%M:%S")
    base_df = pd.DataFrame(index=(ts_start, ts_end)).asfreq("h")

    # Create individual seasonal components of the wind speed
    # Hourly variation, throughout the day
    base_df["h"] = base_df.index.hour
    base_df["ws_h"] = (np.sin(2 * np.pi * (base_df["h"] - delay_h) / 24) + 1) / 2

    # Monthly variation, throughout the year (smoothed by using dayofyear instead of just the month)
    base_df["d"] = base_df.index.dayofyear
    base_df["ws_d"] = (
        np.sin(2 * np.pi * (base_df["d"] - (delay_m * 30)) / 365) + 1
    ) / 2

    # Yearly variation (smoothed by using dayofyear instead of just the year)
    base_df["y"] = base_df.index.dayofyear / 365 + base_df.index.year
    base_df["y"] -= base_df["y"].min()
    base_df["ws_y"] = (np.sin(2 * np.pi * (base_df["y"] - delay_y) / period_y) + 1) / 2

    # Sum all of the seasonality components and add random noise
    base_df["WS"] = (
        base_df["ws_y"] * amplitude_y*2 + base_df["ws_d"] * amplitude_m*2 + base_df["ws_h"] * amplitude_h*2 + mean_ws
    )
    base_df["WS"] += np.random.normal(0, std_noise, (base_df[["WS"]].size))
    base_df["WS"] = base_df["WS"].clip(0)  # Wind speed can't be negative

    return base_df[["WS"]]
