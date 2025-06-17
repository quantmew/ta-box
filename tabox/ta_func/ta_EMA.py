import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import *


def TA_EMA_Lookback(optInTimePeriod: cython.Py_ssize_t) -> cython.Py_ssize_t:
    """TA_EMA_Lookback(optInTimePeriod) -> Py_ssize_t

    EMA Lookback
    """
    return optInTimePeriod - 1


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_INT_EMA(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    optInK_1: cython.double,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """Internal EMA implementation without parameter checks"""
    tempReal: cython.double
    prevMA: cython.double
    i: cython.Py_ssize_t
    today: cython.Py_ssize_t
    outIdx: cython.Py_ssize_t
    lookbackTotal: cython.Py_ssize_t

    lookbackTotal = TA_EMA_Lookback(optInTimePeriod)

    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_SUCCESS

    outBegIdx[0] = startIdx

    # Calculate the first EMA value as the seed
    today = startIdx - lookbackTotal
    i = optInTimePeriod
    tempReal = 0.0
    while i > 0:
        tempReal += inReal[today]
        today += 1
        i -= 1

    prevMA = tempReal / optInTimePeriod

    # Skip the unstable period
    while today <= startIdx:
        prevMA = ((inReal[today] - prevMA) * optInK_1) + prevMA
        today += 1

    # Write the first value
    outReal[0] = prevMA
    outIdx = 1

    # Calculate the remaining range
    while today <= endIdx:
        prevMA = ((inReal[today] - prevMA) * optInK_1) + prevMA
        outReal[outIdx] = prevMA
        outIdx += 1
        today += 1

    outNBElement[0] = outIdx
    return TA_SUCCESS


def TA_EMA(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
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
    k: cython.double = 2.0 / (optInTimePeriod + 1)

    # Call internal implementation
    return TA_INT_EMA(startIdx, endIdx, inReal, optInTimePeriod, k, outBegIdx, outNBElement, outReal)


def EMA(real: np.ndarray, timeperiod: int = 30):
    """EMA(real[, timeperiod=30])

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
    outBegIdx = np.zeros(1, dtype=np.int64)
    outNBElement = np.zeros(1, dtype=np.int64)

    retCode = TA_EMA(0, endIdx, real[startIdx:], timeperiod, outBegIdx, outNBElement, outReal[lookback:])
    if retCode != TA_SUCCESS:
        return outReal
    return outReal
