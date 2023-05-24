import pandas as pd
from utils import compute_eri
import matplotlib.pyplot as plt
from data import tracker_feeder
from portfolio import Performance, EqualWeighted, InverseVol, HRP, MinVar, ERC, MaxSharpe
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Data
df = tracker_feeder()
df = df['br equities'].dropna(how='all')
df = df[['BOVA', 'BBSD']]
eri = compute_eri(df)

cov = eri.pct_change(1).dropna().cov() * 252
corr = eri.pct_change(1).dropna().corr()
print(corr)

perf_asset = Performance(eri)
print(perf_asset.table)

eri.plot(grid=True, title='ERIs')
plt.show()

# ===== Construction Methods =====
# Equal weighted
ew = EqualWeighted(eri)

# Inverse Vol
iv = InverseVol(eri, com=252)

# Hirarchical Risk Parity
hrp = HRP(cov=cov, eri=eri)
hrp.plot_corr_matrix()
hrp.plot_dendrogram()

# Max Sharpe
mu = perf_asset.std * 0.1
ms = MaxSharpe(mu=mu, cov=cov, eri=eri, risk_aversion=2, short_sell=False)
ms.plot()

# ERC
erc = ERC(cov=cov, eri=eri, short_sell=False)

# Min Variance
mv = MinVar(cov=cov, eri=eri, short_sell=False)


# ===== Charts =====
df_eri = pd.concat([ew.eri, iv.eri, hrp.eri, mv.eri, erc.eri, ms.eri], axis=1)

# Performance
perf_port = Performance(df_eri)
print(perf_port.table)

# ERI Chart
df_eri.plot(grid=True, title='ERIs')
plt.show()

# Weights
df_w = pd.concat([ew.weights, iv.weights, hrp.weights, mv.weights, erc.weights, ms.weights], axis=1)
df_w = df_w.fillna(0)
df_w['Mean'] = df_w.mean(axis=1)
df_w = df_w.sort_values('Mean')
ax = df_w.plot(grid=True, title='Weights', kind='bar')
plt.show()
