from data import read_ntnb, tracker_uploader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 250)

# User defined parameters
start_date = '2006-01-01'

# Read the Data - LTN and NTNF
ntnb = read_ntnb()

ntnb = ntnb[ntnb['reference date'] >= start_date]

vna = ntnb.pivot_table(values='vna', index='reference date', aggfunc='mean')
vna = vna.rename({'vna': 'vna ntnb'}, axis=1)

tracker_uploader(vna, 'auxiliar')
