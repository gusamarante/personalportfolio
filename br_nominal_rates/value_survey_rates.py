"""
Computes the expected cumulative realized value of the SELIC rate.
In other words, it builds a zero-coupoun curve based on the expected selic rate from the focus survey
"""

# TODO use standard deviation as a measure of rate uncertainty
# TODO no fundo eu preciso da conta de 1y fwd, 2y fwd, etc.... para todos os indicadores

from data import grab_connection, sgs
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
# curve = metric.iloc[-1].dropna()
# curve.loc['2023-04-25'] = 13.75
# curve = curve.sort_index()
# curve.to_frame('Selic')
# days = (curve.index - pd.to_datetime(pd.to_datetime('today').date())).days
print(df)

