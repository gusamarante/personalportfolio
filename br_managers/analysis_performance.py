from data import tracker_feeder
from finmath import Performance

df = tracker_feeder()
df = df['br managers']

perf = Performance(df)
perf.table.to_clipboard()