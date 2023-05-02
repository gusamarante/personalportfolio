from data import tracker_feeder
from utils import timeseries_momentum, macross

df = tracker_feeder()

df = df['br nominal rates']
vars = df.columns
vars2keep = df.columns.str.contains('y')
df = df[vars[vars2keep]]
df = df.dropna()

macross(df)
