import matplotlib.pyplot as plt
from data import DROPBOX, read_ltn_ntnf, tracker_uploader
from tqdm import tqdm
from time import time
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 250)

# User defined parameters
notional_start = 100
trading_cost = 0 # times the bidask spread
start_date = '2006-01-01'

# Read the Data - LTN and NTNF
ntnf = read_ltn_ntnf()

ntnf = ntnf[ntnf['reference date'] >= start_date]

bonds2loop = list(ntnf['bond code'].unique())
df_trackers = pd.DataFrame()

for bond in bonds2loop:
    # Filter the data
    aux_data = ntnf[ntnf['bond code'] == bond]
    aux_data = aux_data.set_index('reference date')
    aux_data = aux_data[aux_data['price'] != 0]

    # Find the bond name
    mat_date = aux_data['maturity'].max()
    bond_name = 'error'
    if bond[6:9] == 'LTN':
        bond_name = f"LTN {mat_date.year}-{str(mat_date.month).zfill(2)}"
    elif bond[6:9] == 'NTF':
        bond_name = f"NTNF {mat_date.year}-{str(mat_date.month).zfill(2)}"

    # Loop dates to find quantities
    dates2loop = aux_data.index
    aux_data.loc[dates2loop[0], 'quantity'] = 1

    for date, datem1 in tqdm(zip(dates2loop[1:], dates2loop[:-1]), bond_name):
        aux_data.loc[date, 'quantity'] = aux_data.loc[datem1, 'quantity'] * (1 + aux_data.loc[date, 'coupon'] / aux_data.loc[date, 'price'])

    aux_data['notional'] = aux_data['quantity'] * aux_data['price']
    aux_data['tracker'] = 100 * aux_data['notional'] / aux_data['notional'].iloc[0]

    df_trackers = pd.concat([df_trackers, aux_data['tracker'].rename(bond_name)], axis=1)

df_trackers = df_trackers.sort_index()
tracker_uploader(df_trackers, 'br nominal rates')
