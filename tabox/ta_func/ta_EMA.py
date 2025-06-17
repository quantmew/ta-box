import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import *


def TA_EMA_Lookback(optInTimePeriod: cython.Py_ssize_t) -> cython.Py_ssize_t:
    """TA_EMA_Lookback(optInTimePeriod) -> Py_ssize_t

    EMA Lookback
    """
    return optInTimePeriod - 1


def TA_EMA(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.Py_ssize_t,
    outReal: cython.double[::1],
) -> cython.int:
    """TA_EMA - Exponential Moving Average

    Input  = double
    Output = double

    Optional Parameters
    -------------------
    optInTimePeriod:(From 2 to 100000)
       Number of period
    """
    # parameters check
    if startIdx < 0:
        return TA_OUT_OF_RANGE_START_INDEX
    if endIdx < 0 or endIdx < startIdx:
        return TA_OUT_OF_RANGE_END_INDEX
    if optInTimePeriod < 2 or optInTimePeriod > 100000:
        return TA_BAD_PARAM

    # calculate k
    k = 2.0 / (optInTimePeriod + 1)

    # calculate the first EMA value as the seed
    today = startIdx
    tempReal = 0.0
    for i in range(optInTimePeriod):
        tempReal += inReal[today]
        today += 1

    prevMA = tempReal / optInTimePeriod

    # skip the unstable period
    while today <= startIdx:
        prevMA = ((inReal[today] - prevMA) * k) + prevMA
        today += 1

    # write the first value
    outReal[0] = prevMA
    outIdx = 1

    # calculate the remaining range
    while today <= endIdx:
        prevMA = ((inReal[today] - prevMA) * k) + prevMA
        outReal[outIdx] = prevMA
        outIdx += 1
        today += 1

    return TA_SUCCESS


def EMA(real: np.ndarray, timeperiod: int):
    """MA(real[, timeperiod=?, matype=?])

    Exponential Moving average (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    real = check_array(real)
    check_timeperiod(timeperiod)

    length: cython.Py_ssize_t = real.shape[0]

    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_EMA_Lookback(timeperiod)

    outReal = np.full_like(real, np.nan)

    retCode = TA_EMA(0, endIdx, real[startIdx:], timeperiod, outReal[lookback:])
    return outReal
