import cython
import numpy as np
from .ta_utils import check_array, check_begidx1
from ..retcode import *
from .ta_EMA import TA_EMA, TA_EMA_Lookback, TA_INT_EMA

def TA_TEMA_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """TA_TEMA_Lookback(optInTimePeriod) -> Py_ssize_t

    Triple Exponential Moving Average Lookback
    """
    if optInTimePeriod == 0:
        optInTimePeriod = 30
    if optInTimePeriod < 2 or optInTimePeriod > 100000:
        return -1

    return TA_EMA_Lookback(optInTimePeriod) * 3

@cython.boundscheck(False)
@cython.wraparound(False)
def TA_TEMA(
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
    lookbackTotal = lookbackEMA * 3

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
    retCode = TA_INT_EMA(startIdx - (lookbackEMA * 2), endIdx, inReal, optInTimePeriod, k,
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

    # Calculate the third EMA
    outBegIdx3 = np.zeros(1, dtype=np.int64)
    outNbElement3 = np.zeros(1, dtype=np.int64)

    retCode = TA_INT_EMA(0, outNbElement2[0] - 1, secondEMA, optInTimePeriod, k,
                         outBegIdx3, outNbElement3, outReal)
    if retCode != TA_SUCCESS or outNbElement3[0] == 0:
        return retCode

    # Calculate TEMA
    firstEMAIdx = outBegIdx3[0] + outBegIdx2[0]
    secondEMAIdx = outBegIdx3[0]
    for i in range(outNbElement3[0]):
        outReal[i] += (3.0 * firstEMA[firstEMAIdx + i]) - (3.0 * secondEMA[secondEMAIdx + i])

    outBegIdx[0] = firstEMAIdx + outBegIdx1[0]
    outNBElement[0] = outNbElement3[0]

    return TA_SUCCESS

def TEMA(real: np.ndarray, timeperiod: int = 30):
    """TEMA(real, timeperiod=30)

    Triple Exponential Moving Average

    Inputs:
        real: (any ndarray)
        timeperiod: (int) Number of period
    Outputs:
        real: (ndarray) Triple Exponential Moving Average
    """
    real = check_array(real)

    outReal = np.full_like(real, np.nan)
    length: cython.Py_ssize_t = real.shape[0]

    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback = startIdx + TA_TEMA_Lookback(timeperiod)

    outBegIdx: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.int64)
    outNBElement: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.int64)

    TA_TEMA(0, endIdx, real[startIdx:], timeperiod,
            outBegIdx, outNBElement, outReal[lookback:])
    return outReal 