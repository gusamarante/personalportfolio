import numpy as np


def timeseries_momentum(data, com=60, hp=252):
    """
    Get a trend following measure from Moskowitz et al (2011). This function
    is built for daily data. All of the signal are returned with the proper
    information set lags.
    :param data: Return Indexes
    :param com: center of mass of the EWM for the volatility
    :param hp: holding period/lookback paeriod of the returns
    :return: ret: return in the holding period,
             ret_scaled: returns scaled by their ex-ante volatility,
             ret_sign: the direction of the return
    """

    # Ex-ante volatility estimate
    vol = data.pct_change(1).ewm(com=com).std() * np.sqrt(hp)
    vol = vol.shift(1)

    # Return
    ret = data.pct_change(hp)
    ret_scaled = ret / vol
    ret = ret.shift(1)
    ret_scaled = ret_scaled.shift(1)
    ret_sign = np.sign(ret)

    return ret, ret_scaled, ret_sign
