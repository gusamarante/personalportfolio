import pandas as pd
import statsmodels.api as sm
import numpy as np
from data import tracker_feeder, signal_feeder  # TODO Apagar


class Regression(object):
    one_by_one_table = None

    def __init__(self, x, y, one_by_one=False):

        # Align indexes
        new_index = x.dropna().index.intersection(y.dropna().index)
        x = x.reindex(new_index)
        y = y.reindex(new_index)

        self.x = sm.add_constant(x)
        self.y = y
        self.full_results = sm.OLS(self.y, self.x).fit()

        if one_by_one:
            table_obo = pd.DataFrame()
            for num, var in enumerate(x.columns):
                aux_res = sm.OLS(self.y, sm.add_constant(self.x[var])).fit()

                table_obo.loc['const', f'({num + 1})'] = aux_res.params['const']
                table_obo.loc['const p-value', f'({num + 1})'] = aux_res.pvalues['const']

                table_obo.loc[var, f'({num + 1})'] = aux_res.params[var]
                table_obo.loc[f'{var} p-value', f'({num + 1})'] = aux_res.pvalues[var]

                table_obo.loc['R2', f'({num + 1})'] = aux_res.rsquared
                table_obo.loc['N', f'({num + 1})'] = aux_res.nobs
                table_obo.loc['SD(y)', f'({num + 1})'] = np.sqrt(aux_res.mse_total)
                table_obo.loc['SD(resid)', f'({num + 1})'] = np.sqrt(aux_res.mse_resid)

            for var in x.columns:
                table_obo.loc[var, f'({num + 2})'] = self.full_results.params[var]
                table_obo.loc[f'{var} p-value', f'({num + 2})'] = self.full_results.pvalues[var]

            table_obo.loc['const', f'({num + 2})'] = self.full_results.params['const']
            table_obo.loc['const p-value', f'({num + 2})'] = self.full_results.pvalues['const']

            table_obo.loc['R2', f'({num + 2})'] = self.full_results.rsquared
            table_obo.loc['N', f'({num + 2})'] = self.full_results.nobs
            table_obo.loc['SD(y)', f'({num + 2})'] = np.sqrt(self.full_results.mse_total)
            table_obo.loc['SD(resid)', f'({num + 2})'] = np.sqrt(self.full_results.mse_resid)


            hold = table_obo.loc[['R2', 'N', 'SD(y)', 'SD(resid)']]
            table_obo = table_obo.drop(['R2', 'N', 'SD(y)', 'SD(resid)'])
            table_obo = pd.concat([table_obo, hold], axis=0)

            self.one_by_one_table = table_obo








# TODO APAGAR
trackers = tracker_feeder()
trackers = trackers['br nominal rates']['NTNF 8y'].dropna()
trackers = trackers.resample('M').last()
trackers = trackers.pct_change(1)

signals = signal_feeder()
signals = signals['NTNF 8y']
signals = signals.shift(1)
signals = signals.resample('M').last().shift(1)

reg = Regression(signals, trackers, one_by_one=True)
print(reg.one_by_one_table)
reg.one_by_one_table.to_clipboard()
