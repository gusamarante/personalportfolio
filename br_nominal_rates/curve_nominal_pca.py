from data import curve_feeder, sgs, signal_uploader, tracker_feeder
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

df_tracker = tracker_feeder()
df_tracker = df_tracker['br nominal rates']
vars2keep = df_tracker.columns.str.contains('y')
df_tracker = df_tracker[df_tracker.columns[vars2keep]]
df_tracker = df_tracker.dropna()

df = curve_feeder()
df = df['br nominal bonds']
df = df.dropna(how='all')

# Add selic to 0
selic = sgs({432: 1})
selic = (selic - 0.1)/100
df[0] = selic
df = df.sort_index(axis=1)

# Cubic Spline Interpolation
df = df.interpolate(method='cubic', axis=1)
df = df.dropna(how='all', axis=1)

# Select maturities of interest
mats2keepy = [0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8]
mats2keep = [mat*252 for mat in mats2keepy]
df = df[mats2keep]
df = df.dropna(axis=1)

# PCA
pca = PCA(n_components=3)
pca.fit(df.values)

df_var_full = pd.Series(data=pca.explained_variance_ratio_,
                        index=['PC 1', 'PC 2', 'PC 3'])
df_loadings_full = pd.DataFrame(data=pca.components_.T,
                                columns=['PC 1', 'PC 2', 'PC 3'],
                                index=df.columns)
df_mean_full = pd.DataFrame(data=pca.mean_, index=df.columns,
                            columns=['Médias'])
df_pca_full = pd.DataFrame(data=pca.transform(df.values),
                           columns=['PC 1', 'PC 2', 'PC 3'],
                           index=df.index)

signal = np.sign(df_loadings_full.iloc[-1])
df_loadings_full = df_loadings_full * signal
df_pca_full = df_pca_full * signal

df_signal = df_pca_full.melt(ignore_index=False)
df_signal = df_signal.reset_index(names='refdate')
df_signal = df_signal.rename({'variable': 'signal_name'}, axis=1)
df_signal['signal_family'] = 'value'
df_signal['pillar'] = 'br nominal bonds'

for bond in df_tracker.columns:
    aux = df_signal.copy()
    aux['asset'] = bond
    signal_uploader(aux)

