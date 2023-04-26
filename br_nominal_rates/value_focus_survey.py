"""
Computes measures from the focus survey, which can be used as value measures
"""

from data import grab_connection, sgs, curve_uploader
from calendars import DayCounts
from tqdm import tqdm
import pandas as pd

dc = DayCounts('BUS/252', calendar='anbima')
conn = grab_connection()

# Grab Focus Survey Data
query = 'SELECT * FROM focus WHERE indicador = "selic"'
df = pd.read_sql(sql=query, con=conn)

df = df[df['basecalculo']==0]  # Normal survey, not the last 5 days
df['data'] = pd.to_datetime(df['data'])
df['datareferencia'] = pd.to_datetime(df['datareferencia'])

# Grab Selic Data
selic = sgs({432: 'selic'})

# Build the curve based on survey rates
metric = df.pivot(index='data', columns='datareferencia', values='mediana')
focus_curve = pd.DataFrame(columns=range(252*10))

for date in tqdm(metric.index, 'Focus Selic Median'):
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
    curve['DU'] = dc.days(date, curve.index)
    curve = curve.set_index('DU')

    focus_curve.loc[date] = curve['survey curve']

focus_curve = focus_curve.melt(ignore_index=False, var_name='daycount', value_name='yield')
focus_curve = focus_curve.dropna()
focus_curve = focus_curve.reset_index()
focus_curve = focus_curve.rename({'index': 'refdate'}, axis=1)
focus_curve['curvename'] = 'focus selic median'

curve_uploader(focus_curve)


# Build the standard deviation curve
metric = df.pivot(index='data', columns='datareferencia', values='desviopadrao')
focus_curve = pd.DataFrame(columns=range(252*10))

for date in tqdm(metric.index, 'Focus Selic SD'):
    curve = metric.loc[date].dropna()
    curve = curve.sort_index()
    curve = curve.to_frame('sd selic')
    curve['DU'] = dc.days(date, curve.index)
    curve = curve.set_index('DU')

    focus_curve.loc[date] = curve['sd selic']

focus_curve = focus_curve.melt(ignore_index=False, var_name='daycount', value_name='yield')
focus_curve = focus_curve.dropna()
focus_curve = focus_curve.reset_index()
focus_curve = focus_curve.rename({'index': 'refdate'}, axis=1)
focus_curve['curvename'] = 'focus selic sd'

curve_uploader(focus_curve)
