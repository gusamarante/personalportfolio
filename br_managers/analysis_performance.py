from data import tracker_feeder, sgs
from portfolio import Performance

cdi = sgs({12: 'CDI'})
cdi = cdi['CDI']/100

df = tracker_feeder()
df = df['br managers']
df_ret = df.pct_change(1)
df_ret = df_ret.iloc[:-3]
df_er = df_ret.subtract(cdi, axis=0).dropna(how='all')
df_er = (1 + df_er).cumprod()
df_er = df_er[df_er.index >= '2010-01-01']

perf = Performance(df_er)
perf.table.to_clipboard()
