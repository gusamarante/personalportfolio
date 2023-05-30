import pandas as pd
from data import tracker_feeder
from utils import compute_eri
from portfolio import Performance, EqualWeighted, InverseVol, HRP, ERC, MaxSharpe, MinVar
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

trackers = tracker_feeder()

br_equities = trackers['br equities']['BOVA'].dropna()
br_nominal = trackers['br nominal rates']['NTNF 8y'].dropna()
br_real = trackers['br real rates']['NTNB 10y'].dropna()
br_credit = trackers['br credit']['IDA Geral'].dropna()
gl_equities = trackers['global equities']['IVVB'].dropna()

tri = pd.concat([br_equities, br_nominal, br_real, br_credit, gl_equities], axis=1)

eri = compute_eri(tri)
perf_asset = Performance(eri)
print(perf_asset.table)

corr = eri.pct_change(1).dropna().corr()
cov = eri.pct_change(1).dropna().cov() * 252

eri.plot()
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
mu = perf_asset.std * 0.15
ms = MaxSharpe(mu=mu, cov=cov, eri=eri, risk_aversion=2, short_sell=False)
ms.plot()

# ERC
erc = ERC(cov=cov, eri=eri, short_sell=False)

# Min Variance
mv = MinVar(cov=cov, eri=eri, short_sell=False)


# ===== Charts =====
port_eri = pd.concat([ew.eri, iv.eri, hrp.eri, mv.eri, erc.eri, ms.eri], axis=1)

# Performance
perf_port = Performance(port_eri)
print(perf_port.table)

# ERI Chart
port_eri.plot(grid=True, title='ERIs')
plt.show()

# Weights
df_w = pd.concat([ew.weights, iv.weights, hrp.weights, mv.weights, erc.weights, ms.weights], axis=1)
df_w = df_w.fillna(0)
df_w['Mean'] = df_w.mean(axis=1)
df_w = df_w.sort_values('Mean')
ax = df_w.plot(grid=True, title='Weights', kind='bar')
plt.show()
