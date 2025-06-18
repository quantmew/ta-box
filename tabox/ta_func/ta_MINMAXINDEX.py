import cython
from cython.parallel import prange
import numpy as np
from .ta_utils import check_array, check_begidx1
from ..retcode import TA_RetCode


def TA_MINMAXINDEX_Lookback(optInTimePeriod: cython.Py_ssize_t) -> cython.Py_ssize_t:
    return optInTimePeriod - 1


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_MINMAXINDEX(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outMinIdx: cython.Py_ssize_t[::1],
    outMaxIdx: cython.Py_ssize_t[::1],
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
    minIdx: cython.Py_ssize_t
    maxIdx: cython.Py_ssize_t
    minValue: cython.double
    maxValue: cython.double

    # Calculate the minimum and maximum value indices
    for i in range(startIdx + optInTimePeriod - 1, endIdx + 1):
        minIdx = i - optInTimePeriod + 1
        maxIdx = minIdx
        minValue = inReal[minIdx]
        maxValue = minValue
        for j in range(minIdx + 1, i + 1):
            if inReal[j] < minValue:
                minValue = inReal[j]
                minIdx = j
            if inReal[j] > maxValue:
                maxValue = inReal[j]
                maxIdx = j
        outMinIdx[outIdx] = minIdx
        outMaxIdx[outIdx] = maxIdx
        outIdx += 1

    outBegIdx[0] = startIdx + optInTimePeriod - 1
    outNBElement[0] = outIdx
    
    return TA_RetCode.TA_SUCCESS


def MINMAXINDEX(real: np.ndarray, timeperiod: cython.Py_ssize_t = 30):
    """MINMAXINDEX(real[, timeperiod=30])

    Indexes of lowest and highest values over a specified period (Math Transform)

    Inputs:
        real: (any ndarray)
        timeperiod: (int) Number of period
    Outputs:
        minidx
        maxidx
    """
    real = check_array(real)

    outMinIdx = np.full_like(real, np.nan, dtype=np.int64)
    outMaxIdx = np.full_like(real, np.nan, dtype=np.int64)
    length: cython.Py_ssize_t = real.shape[0]

    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback = startIdx + TA_MINMAXINDEX_Lookback(timeperiod)

    outBegIdx: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.int64)
    outNBElement: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.int64)

    TA_MINMAXINDEX(0, endIdx, real[startIdx:], timeperiod, outBegIdx, outNBElement, outMinIdx[lookback:], outMaxIdx[lookback:])
    return outMinIdx, outMaxIdx 