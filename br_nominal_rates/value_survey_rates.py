"""
Computes the expected cumulative realized value of the SELIC rate.
In other words, it builds a zero-coupoun curve based on the expected selic rate from the focus survey
"""

# TODO use standard deviation as a measure of rate uncertainty
# TODO no fundo eu preciso da conta de 1y fwd, 2y fwd, etc.... para todos os indicadores

from data import grab_connection, sgs
from tqdm import tqdm
import pandas as pd


# Grab Focus Survey Data
conn = grab_connection()
query = 'SELECT * FROM focus WHERE indicador = "selic"'
df = pd.read_sql(sql=query, con=conn)

df = df[df['basecalculo']==0]  # Normal survey, not the last 5 days
df['data'] = pd.to_datetime(df['data'])
df['datareferencia'] = pd.to_datetime(df['datareferencia'])

# Grab Selic Data
selic = sgs({432: 'selic'})

# Build the curve for one day
metric = df.pivot(index='data', columns='datareferencia', values='mediana')

for date in tqdm(metric.index, 'Generating Curve based on Selic'):
    curve = metric.loc[date].dropna()
    curve.loc[date] = selic.loc[date, 'selic']
    curve = curve.sort_index()
    curve = curve.to_frame('selic')
    curve['days'] = (curve.index - date).days
    curve['ddays'] = curve['days'].diff().fillna(0)
    curve['factors'] = (1 + curve['selic'] / 100) ** (curve['ddays'] / 365.25)
    curve['cumfactors'] = curve['factors'].cumprod()
    curve['survey curve'] = curve['cumfactors'] ** (365.25 / curve['days']) * 100 - 100
    curve.loc[date, 'survey curve'] = selic.loc[date, 'selic']
    curve = curve[['survey curve']]
