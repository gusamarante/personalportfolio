import pandas as pd
from data import tracker_feeder
from utils import compute_eri
import matplotlib.pyplot as plt
from portfolio import EqualWeighted, InverseVol, HRP
import numpy as np

df = tracker_feeder()
df = df['br equities'].dropna(how='all')

eri = compute_eri(df)

# Equal weighted
ew = EqualWeighted(eri)

# Inverse Vol
iv = InverseVol(eri, com=252)

# Hirarchical Risk Parity
cov = eri.pct_change(1).dropna().cov()
hrp = HRP(eri, cov)
hrp.plot_corr_matrix()
hrp.plot_dendrogram()

# Max Sharpe

# ERC

# Charts
df_eri = pd.concat([ew.eri, iv.eri, hrp.eri], axis=1)
df_eri.plot(grid=True, title='ERIs')
plt.show()

df_w = pd.concat([ew.weights, iv.weights, hrp.weights], axis=1)
df_w = df_w.fillna(0)
df_w['Mean'] = df_w.mean(axis=1)
df_w = df_w.sort_values('Mean')

ax = df_w.plot(grid=True, title='Weights', kind='bar')
plt.show()
