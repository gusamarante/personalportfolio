import pandas as pd
from data import tracker_feeder
from utils import compute_eri
import matplotlib.pyplot as plt
from portfolio import inverse_vol, equal_weighted
import numpy as np

df = tracker_feeder()
df = df['br equities'].dropna(how='all')

eri = compute_eri(df)

# Equal weighted
eri_ew, w_ew = equal_weighted(eri)  # TODO rethink how to report weights - TS, static, latest?

# Inverse Vol
eri_iv, w_iv = inverse_vol(eri)


df_eri = pd.concat([eri_ew, eri_iv], axis=1)
df_eri.plot(grid=True, title='ERIs')
plt.show()

df_w = pd.concat([w_ew.iloc[-1].rename('EW'),
                    w_iv.iloc[-1].rename('IV')], axis=1)
df_w = df_w.fillna(0)
ax = df_w.plot(grid=True, title='Weights', kind='bar')
df_w['Mean'] = df_w.mean(axis=1)
ax = ax.plot(df_w['Mean'], color='black')
plt.show()
