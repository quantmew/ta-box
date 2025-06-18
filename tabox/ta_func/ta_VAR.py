import cython
import numpy as np
from .ta_utils import check_array, check_begidx1, check_timeperiod
from ..retcode import TA_RetCode

def TA_VAR_Lookback(optInTimePeriod: cython.int, optInNbDev: cython.double) -> cython.Py_ssize_t:
    """TA_VAR_Lookback(optInTimePeriod, optInNbDev) -> Py_ssize_t

    Variance Lookback
    """
    if optInTimePeriod == 0:
        optInTimePeriod = 5
    elif optInTimePeriod < 1 or optInTimePeriod > 100000:
        return -1

    if optInNbDev == 0:
        optInNbDev = 1.0
    elif optInNbDev < -3.0e37 or optInNbDev > 3.0e37:
        return -1

    return optInTimePeriod - 1

@cython.boundscheck(False)
@cython.wraparound(False)
def TA_INT_VAR(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """Internal implementation of Variance calculation"""
    nbInitialElementNeeded: cython.Py_ssize_t = optInTimePeriod - 1

    if startIdx < nbInitialElementNeeded:
        startIdx = nbInitialElementNeeded

    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    periodTotal1: cython.double = 0.0
    periodTotal2: cython.double = 0.0
    trailingIdx: cython.Py_ssize_t = startIdx - nbInitialElementNeeded

    i: cython.Py_ssize_t = trailingIdx
    if optInTimePeriod > 1:
        while i < startIdx:
            tempReal: cython.double = inReal[i]
            periodTotal1 += tempReal
            tempReal *= tempReal
            periodTotal2 += tempReal
            i += 1

    outIdx: cython.Py_ssize_t = 0
    while i <= endIdx:
        tempReal: cython.double = inReal[i]
        periodTotal1 += tempReal
        tempReal *= tempReal
        periodTotal2 += tempReal

        meanValue1: cython.double = periodTotal1 / optInTimePeriod
        meanValue2: cython.double = periodTotal2 / optInTimePeriod

        tempReal: cython.double = inReal[trailingIdx]
        periodTotal1 -= tempReal
        tempReal *= tempReal
        periodTotal2 -= tempReal

        outReal[outIdx] = meanValue2 - meanValue1 * meanValue1
        outIdx += 1
        i += 1
        trailingIdx += 1

    outNBElement[0] = outIdx
    outBegIdx[0] = startIdx

    return TA_RetCode.TA_SUCCESS

@cython.boundscheck(False)
@cython.wraparound(False)
def TA_VAR(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    optInNbDev: cython.double,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    # Parameters check
    if startIdx < 0:
        return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
    if endIdx < 0 or endIdx < startIdx:
        return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX

    if optInTimePeriod == 0:
        optInTimePeriod = 5
    elif optInTimePeriod < 1 or optInTimePeriod > 100000:
        return TA_RetCode.TA_BAD_PARAM

    if optInNbDev == 0:
        optInNbDev = 1.0
    elif optInNbDev < -3.0e37 or optInNbDev > 3.0e37:
        return TA_RetCode.TA_BAD_PARAM

    return TA_INT_VAR(startIdx, endIdx, inReal, optInTimePeriod, outBegIdx, outNBElement, outReal)

def VAR(real: np.ndarray, timeperiod: int = 5, nbdev: float = 1.0):
    """VAR(real[, timeperiod=5, nbdev=1.0])

    Variance (Statistic Functions)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 5
        nbdev: 1.0
    Outputs:
        real
    """
    real = check_array(real)
    check_timeperiod(timeperiod)

    outReal = np.full_like(real, np.nan)
    length: cython.Py_ssize_t = real.shape[0]

    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback = startIdx + TA_VAR_Lookback(timeperiod, nbdev)

    outBegIdx: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    outNBElement: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)

    TA_VAR(0, endIdx, real[startIdx:], timeperiod, nbdev,
           outBegIdx, outNBElement, outReal[lookback:])
    return outReal 