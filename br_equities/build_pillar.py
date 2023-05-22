import pandas as pd
from utils import compute_eri
import matplotlib.pyplot as plt
from data import tracker_feeder
from portfolio import Performance, EqualWeighted, InverseVol, HRP, MinVar, ERC

# Data
df = tracker_feeder()
df = df['br equities'].dropna(how='all')
eri = compute_eri(df)

#  ===== Construction Methods =====
# Equal weighted
ew = EqualWeighted(eri)

# Inverse Vol
iv = InverseVol(eri, com=252)

# Hirarchical Risk Parity
cov = eri.pct_change(1).dropna().cov()
hrp = HRP(eri)
# hrp.plot_corr_matrix()
# hrp.plot_dendrogram()

# Max Sharpe

# ERC
erc = ERC(eri, short_sell=False)

# Min Variance
mv = MinVar(eri, short_sell=False)


# ===== Charts =====
df_eri = pd.concat([ew.eri, iv.eri, hrp.eri, mv.eri, erc.eri], axis=1)

# Performance
perf = Performance(df_eri)
print(perf.table)

# ERI
df_eri.plot(grid=True, title='ERIs')
plt.show()

# Weights
df_w = pd.concat([ew.weights, iv.weights, hrp.weights, mv.weights, erc.weights], axis=1)
df_w = df_w.fillna(0)
df_w['Mean'] = df_w.mean(axis=1)
df_w = df_w.sort_values('Mean')
ax = df_w.plot(grid=True, title='Weights', kind='bar')
plt.show()
