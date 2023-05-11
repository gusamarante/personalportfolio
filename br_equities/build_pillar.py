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
ew = equal_weighted(eri)

# Inverse Vol
inv_vol = inverse_vol(eri)


df2plot = pd.concat([ew, inv_vol], axis=1)
df2plot.plot(grid=True)
plt.show()
