import cython
from cython.parallel import prange
import numpy as np
from .ta_utils import check_array, check_begidx1
from ..retcode import TA_RetCode


def TA_MINMAX_Lookback(optInTimePeriod: cython.Py_ssize_t) -> cython.Py_ssize_t:
    return optInTimePeriod - 1


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_MINMAX(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outMin: cython.double[::1],
    outMax: cython.double[::1],
) -> cython.int:
    # Parameters check
    if startIdx < 0:
        return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
    if endIdx < 0 or endIdx < startIdx:
        return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
    if optInTimePeriod < 2:
        return TA_RetCode.TA_BAD_PARAM
    
    outIdx: cython.Py_ssize_t = 0
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    minValue: cython.double
    maxValue: cython.double

    # Calculate the minimum and maximum values
    for i in range(startIdx + optInTimePeriod - 1, endIdx + 1):
        minValue = inReal[i - optInTimePeriod + 1]
        maxValue = minValue
        for j in range(i - optInTimePeriod + 2, i + 1):
            if inReal[j] < minValue:
                minValue = inReal[j]
            if inReal[j] > maxValue:
                maxValue = inReal[j]
        outMin[outIdx] = minValue
        outMax[outIdx] = maxValue
        outIdx += 1

    outBegIdx[0] = startIdx + optInTimePeriod - 1
    outNBElement[0] = outIdx
    
    return TA_RetCode.TA_SUCCESS


def MINMAX(real: np.ndarray, timeperiod: cython.Py_ssize_t = 30):
    """MINMAX(real[, timeperiod=30])

    Lowest and highest values over a specified period (Math Transform)

    Inputs:
        real: (any ndarray)
        timeperiod: (int) Number of period
    Outputs:
        min
        max
    """
    real = check_array(real)

    outMin = np.full_like(real, np.nan)
    outMax = np.full_like(real, np.nan)
    length: cython.Py_ssize_t = real.shape[0]

    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback = startIdx + TA_MINMAX_Lookback(timeperiod)

    outBegIdx: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.int64)
    outNBElement: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.int64)

    TA_MINMAX(0, endIdx, real[startIdx:], timeperiod, outBegIdx, outNBElement, outMin[lookback:], outMax[lookback:])
    return outMin, outMax 