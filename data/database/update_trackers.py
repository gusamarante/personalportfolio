import pandas as pd
from data.external_sheets import read_etf

# BR Equities
br_equities = read_etf(['BOVA', 'BCIC', 'BDEF', 'BBSD', 'BREW', 'BRAX', 'BMMT', 'SMAL'])
br_equities = br_equities.melt(ignore_index=False).reset_index()
br_equities = br_equities.rename({'index': 'date',
                                  'variable': 'asset'},
                                 axis=1)
br_equities = br_equities.dropna()
br_equities['pillar'] = 'br equities'

# TODO delete existing from SQL table
# TODO Update trackers to the table
