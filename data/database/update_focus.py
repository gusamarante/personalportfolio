from data import focus_scrapper, grab_connection

conn = grab_connection()

# IPCA Mensal
df = focus_scrapper('IPCA Mensal')
df.columns = df.columns.str.lower()
df = df.dropna()
for var in ['indicador', 'frequencia']:
    df[var] = df[var].str.lower()
df.to_sql('focus', con=conn, index=False, if_exists='replace')


# IPCA Anual
df = focus_scrapper('IPCA Anual')
df.columns = df.columns.str.lower()
df = df.dropna()
for var in ['indicador', 'frequencia']:
    df[var] = df[var].str.lower()
df.to_sql('focus', con=conn, index=False, if_exists='append')


# SELIC Anual
df = focus_scrapper('SELIC Anual')
df.columns = df.columns.str.lower()
df = df.dropna()
for var in ['indicador', 'frequencia']:
    df[var] = df[var].str.lower()
df.to_sql('focus', con=conn, index=False, if_exists='append')
