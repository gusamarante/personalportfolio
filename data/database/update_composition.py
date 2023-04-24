# TODO Read the current portfolio composition from excel and save it to the database
# TODO Save the current portfolio weights (actual weights), current porfolio values (actual values)

import pandas as pd
from data import DROPBOX

cols = [5, 6, 7, 8, 9]
df = pd.read_excel(DROPBOX.joinpath('Personal Portfolio.xlsx'),
                   usecols=cols)
df = df.dropna()

print(df)
