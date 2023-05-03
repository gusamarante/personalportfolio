"""
This is not finished nearly finished. I just started
"""

from data import tracker_feeder, sgs
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

df = tracker_feeder()
df = df['br managers']

ret = df.resample('M').last().pct_change(1).tail(10*12).dropna(how='all').dropna(axis=1)

pca = PCA(n_components=ret.shape[1])
pca.fit(ret.values)

df_var_full = pd.Series(data=pca.explained_variance_ratio_,
                        index=[f'PC {x+1}' for x in range(ret.shape[1])])
df_loadings_full = pd.DataFrame(data=pca.components_.T,
                                columns=[f'PC {x+1}' for x in range(ret.shape[1])],
                                index=ret.columns)
df_mean_full = pd.DataFrame(data=pca.mean_, index=ret.columns,
                            columns=['Médias'])
df_pca_full = pd.DataFrame(data=pca.transform(ret.values),
                           columns=[f'PC {x+1}' for x in range(ret.shape[1])],
                           index=ret.index)

signal = np.sign(df_loadings_full.iloc[-1])
df_loadings_full = df_loadings_full * signal
df_pca_full = df_pca_full * signal

a = 1