import cython
import numpy as np
from .ta_utils import check_array, check_begidx1
from ..retcode import *
from .ta_EMA import TA_EMA, TA_EMA_Lookback, TA_INT_EMA

def TA_DEMA_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """TA_DEMA_Lookback(optInTimePeriod) -> Py_ssize_t

    Double Exponential Moving Average Lookback
    """
    if optInTimePeriod == 0:
        optInTimePeriod = 30
    if optInTimePeriod < 2 or optInTimePeriod > 100000:
        return -1

    return TA_EMA_Lookback(optInTimePeriod) * 2

@cython.boundscheck(False)
@cython.wraparound(False)
def TA_DEMA(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    # Parameters check
    if startIdx < 0:
        return TA_OUT_OF_RANGE_START_INDEX
    if endIdx < 0 or endIdx < startIdx:
        return TA_OUT_OF_RANGE_END_INDEX

    if optInTimePeriod == 0:
        optInTimePeriod = 30
    elif optInTimePeriod < 2 or optInTimePeriod > 100000:
        return TA_BAD_PARAM

    lookbackEMA = TA_EMA_Lookback(optInTimePeriod)
    lookbackTotal = lookbackEMA * 2

    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_SUCCESS

    tempInteger = lookbackTotal + (endIdx - startIdx) + 1
    firstEMA = np.zeros(tempInteger, dtype=np.float64)

    k = 2.0 / (optInTimePeriod + 1)
    outBegIdx1 = np.zeros(1, dtype=np.int64)
    outNbElement1 = np.zeros(1, dtype=np.int64)

    # Calculate the first EMA
    retCode = TA_INT_EMA(startIdx - lookbackEMA, endIdx, inReal, optInTimePeriod, k,
                         outBegIdx1, outNbElement1, firstEMA)
    if retCode != TA_SUCCESS or outNbElement1[0] == 0:
        return retCode

    # Calculate the second EMA
    secondEMA = np.zeros(outNbElement1[0], dtype=np.float64)
    outBegIdx2 = np.zeros(1, dtype=np.int64)
    outNbElement2 = np.zeros(1, dtype=np.int64)

    retCode = TA_INT_EMA(0, outNbElement1[0] - 1, firstEMA, optInTimePeriod, k,
                         outBegIdx2, outNbElement2, secondEMA)
    if retCode != TA_SUCCESS or outNbElement2[0] == 0:
        return retCode

    # Calculate DEMA
    firstEMAIdx = outBegIdx2[0]
    for i in range(outNbElement2[0]):
        outReal[i] = 2.0 * firstEMA[firstEMAIdx + i] - secondEMA[i]

    outBegIdx[0] = outBegIdx1[0] + outBegIdx2[0]
    outNBElement[0] = outNbElement2[0]

    return TA_SUCCESS

def DEMA(real: np.ndarray, timeperiod: int = 30):
    """DEMA(real, timeperiod=30)

    Double Exponential Moving Average

    Inputs:
        real: (any ndarray)
        timeperiod: (int) Number of period
    Outputs:
        real: (ndarray) Double Exponential Moving Average
    """
    real = check_array(real)

    outReal = np.full_like(real, np.nan)
    length: cython.Py_ssize_t = real.shape[0]

    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback = startIdx + TA_DEMA_Lookback(timeperiod)

    outBegIdx: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.int64)
    outNBElement: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.int64)

    TA_DEMA(0, endIdx, real[startIdx:], timeperiod,
            outBegIdx, outNBElement, outReal[lookback:])
    return outReal 