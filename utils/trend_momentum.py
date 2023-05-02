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


def macross(data, window_fast=21, window_slow=252, window_type='rolling'):
    """
    I have a hard time seeing this as a measure of momentum. Intuitively, this looks more lake a value measure
    """

    if window_type == 'rolling':
        ma_fast = data.rolling(window_fast).mean()
        ma_slow = data.rolling(window_slow).mean()
    elif window_type == 'ewm':
        ma_fast = data.ewm(com=window_fast).mean()
        ma_slow = data.ewm(com=window_slow).mean()
    else:
        raise AssertionError("window method not implemented")

    signal = ma_slow / ma_fast - 1
    return signal
