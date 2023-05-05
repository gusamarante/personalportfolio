from data import read_ltn_ntnf, tracker_uploader, curve_uploader
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from calendars import DayCounts
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# User defined parameters
notional_start = 100
trading_cost = 0 # times the bidask spread
start_date = '2006-01-01'

dc = DayCounts(dc='bus/252', calendar='anbima')

# ===== Custom Functions =====
def cashflows_ntnf(reference_date, maturity_date):
    """
    NTN-F paga cupom em 1 de janeiro e 1 de julho. 10% do valor de face
    """

    dates = pd.date_range(start='1980-01-01', end=maturity_date, freq='6MS')
    dates = dates[dates >= reference_date]
    dates = dc.following(dates)
    n_cf = len(dates)
    dus = dc.days(reference_date, dates)

    coupons = 1.1 ** 0.5 - 1 * np.ones(n_cf)
    coupons[-1] = coupons[-1] + 1
    coupons = coupons * 1000

    cashflows = pd.DataFrame(index=dus, data={'cashflow': coupons})

    return cashflows


def cashflows_ltn(reference_date, maturity_date):
    """
    LTN é Zero Coupon
    """
    dus = dc.days(reference_date, maturity_date)
    cashflows = pd.DataFrame(index=[dus], data={'cashflow': [1000]})
    return cashflows


def bootstrapp(cashflows, bond_prices):
    """
    Bootstraps the bonds to find the curve
    """

    # Find the DUs that we can change
    du_dof = cashflows.idxmax().unique()

    def objective_function(disc):
        du = np.insert(du_dof, 0, 0)  # add the first value, which will be fixed at zero
        disc = np.insert(disc, 0, 1)  # add the first value, which will be fixed at one
        interp_obj = interp1d(du, np.log(disc), kind='linear')  # Interpolation of the log of disccounts
        disc = pd.Series(index=cashflows.index, data=np.exp(interp_obj(cashflows.index)))  # Populate the discounts to a series
        sum_dcf = cashflows.multiply(disc, axis=0).sum()  # get the sum of discounted cashflows
        erros = bond_prices.subtract(sum_dcf, axis=0)  # Difference between actual prices and sum of DCF
        erro_total = (erros ** 2).sum() * 1000 # Sum of squarred errors

        try:
            erro_total = erro_total.values[0]
        except AttributeError:
            erro_total = erro_total

        return erro_total

    # Run optimization
    # Initial guess for the vector of disccounts
    init_discount = 0.8 * np.ones(len(du_dof))
    res = minimize(fun=objective_function,
                   x0=init_discount,
                   method=None,
                   options={'disp': False})

    dus = np.insert(du_dof, 0, 0)  # add the first value, which will be fixed at zero
    discount = np.insert(res.x, 0, 1)  # add the first value, which will be fixed at one
    interp_boot = interp1d(dus, np.log(discount))  # Interpolation of the log of disccounts
    discount = pd.Series(index=cashflows.index, data=np.exp(interp_boot(cashflows.index)))

    curve = (1 / discount) ** (252 / discount.index) - 1

    return curve

# Read the Data - LTN and NTNF
ntnf = read_ltn_ntnf()

ntnf = ntnf[ntnf['reference date'] >= start_date]
dates2loop = pd.to_datetime(ntnf['reference date'].unique())
# dates2loop = dates2loop[dates2loop >= start_date]
ano = 2023
dates2loop = dates2loop[dates2loop >= f'{ano}-01-01']
dates2loop = dates2loop[dates2loop <= f'{ano}-12-31']

df_yield_curve = pd.DataFrame()

for today in tqdm(dates2loop, 'Bootstrapping Nominal Bonds'):

    current_bonds = ntnf[ntnf['reference date'] == today].sort_values('du')
    current_bonds = current_bonds.set_index('bond code')

    df_cashflows = pd.DataFrame()

    for bond in current_bonds.index:

        if bond[6:9] == 'LTN':
            aux = cashflows_ltn(reference_date=today,
                                 maturity_date=current_bonds.loc[bond, 'maturity'])

        elif bond[6:9] == 'NTF':
            aux = cashflows_ntnf(reference_date=today,
                                 maturity_date=current_bonds.loc[bond, 'maturity'])

        else:
            raise FileNotFoundError('Wrong Bond Code')

        aux = aux.rename({'cashflow': current_bonds.loc[bond, 'maturity']}, axis=1)
        df_cashflows = pd.concat([df_cashflows, aux], axis=1)

    df_cashflows = df_cashflows.sort_index()
    df_cashflows = df_cashflows.fillna(0)
    prices = current_bonds[['maturity', 'price']].set_index('maturity')

    yield_curve = bootstrapp(df_cashflows, prices)

    yield_curve = yield_curve.to_frame(today)
    yield_curve = yield_curve.melt(ignore_index=False).reset_index()
    yield_curve = yield_curve.rename({'index': 'daycount',
                                      'variable': 'refdate',
                                      'value': 'yield'},
                                     axis=1)

    df_yield_curve = pd.concat([df_yield_curve, yield_curve], axis=0)

df_yield_curve = df_yield_curve.dropna()
df_yield_curve['curvename'] = 'br nominal bonds'

curve_uploader(df_yield_curve, delete_first=True, year=ano)
