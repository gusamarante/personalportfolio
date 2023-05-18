import numpy as np


def equal_weighted(eri):
    """
    - made for daily data
    - assumes daily rebalance
    """

    ret = eri.pct_change(1).dropna(how='all')
    weights = (~ret.isna() * 1).div(ret.count(axis=1), axis=0)
    ew = ret * weights
    ew = ew.sum(axis=1)
    ew = (1 + ew).cumprod()
    ew = 100 * ew / ew.iloc[0]
    ew = ew.rename('Equal Weighted')

    return ew, weights


def inverse_vol(eri, com=252):
    """
    - Made for daily data
    - Outputs assumes daily rebalance
    """

    # Weights
    ret = eri.pct_change(1)
    vols = ret.ewm(com=com, min_periods=com).std() * np.sqrt(252)
    weights = 1 / vols
    weights = weights.div(weights.sum(axis=1), axis=0)

    inv_vol = ret * weights
    inv_vol = inv_vol.dropna(how='all')
    inv_vol = inv_vol.sum(axis=1)
    inv_vol = (1 + inv_vol).cumprod()
    inv_vol = 100 * inv_vol / inv_vol.iloc[0]
    inv_vol = inv_vol.rename('Inverse Vol')

    return inv_vol, weights



