import numpy as np
from data import sgs
import matplotlib.pyplot as plt

start_date = '2003-01-01'
sgs_dict = {
    24364: 'ibcbr',
    432: 'selic',
    433: 'ipca',
}
df_sgs = sgs(sgs_dict)
df_sgs = df_sgs.resample('M').last()

df_sgs['ibcbr'] = df_sgs['ibcbr'].pct_change(3, fill_method=None) * 100
df_sgs['selic'] = df_sgs['selic'].diff(3)
df_sgs['ipca'] = ((1 + df_sgs['ipca']/100).rolling(3).apply(np.prod) - 1) * 100
df_sgs = df_sgs[df_sgs.index >= start_date]

df_sgs.plot()
plt.show()
