from data import tracker_feeder
from models import timeseries_momentum

df = tracker_feeder()
df = df['br nominal rates']
vars2keep = df.columns.str.contains('y')
df = df[df.columns[vars2keep]]
df = df.dropna()

ts_signal, ts_signal_scaled, ts_signal_sign = timeseries_momentum(df)
ts_signal = ts_signal.resample('M').last().shift(1)
ts_signal_scaled = ts_signal_scaled.resample('M').last().shift(1)
ts_signal_sign = ts_signal_sign.resample('M').last().shift(1)

returns = df.pct_change(21).resample('M').last()


