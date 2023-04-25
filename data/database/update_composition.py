import pandas as pd
from data import DROPBOX, grab_connection

cols = [5, 6, 7, 8, 9]
df = pd.read_excel(DROPBOX.joinpath('Personal Portfolio.xlsx'),
                   usecols=cols)
df = df.dropna()
df = df.melt(id_vars=['Date', 'Pillar', 'Asset'], var_name='type')
df.columns = df.columns.str.lower()
for var in ['pillar', 'asset', 'type']:
    df[var] = df[var].str.lower()


conn = grab_connection()
df.to_sql('portfolio_composition', con=conn, index=False, if_exists='append')
