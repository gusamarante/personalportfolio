import pandas as pd
from utils import compute_eri
from data import tracker_feeder
import matplotlib.pyplot as plt
from models import timeseries_momentum, Regression
from portfolio import Performance
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = tracker_feeder()
df = df['br nominal rates']
df = df.dropna()

lookback = [1, 3, 6, 12, 24, 36, 60]
holding_period = [1, 2, 3, 6, 12, 24]

eri = compute_eri(df)

ts_signal_1y, ts_signal_scaled_1y, ts_signal_sign_1y = timeseries_momentum(eri['NTNF 8y'], hp=252)
ts_signal_6m, ts_signal_scaled_6m, ts_signal_sign_6m = timeseries_momentum(eri['NTNF 8y'], hp=126)

# TODO Upload signals

