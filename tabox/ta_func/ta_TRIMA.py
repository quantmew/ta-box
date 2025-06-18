import cython
import numpy as np
from .ta_utils import check_array, check_begidx1
from ..retcode import TA_RetCode

def TA_TRIMA_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """TA_TRIMA_Lookback(optInTimePeriod) -> Py_ssize_t

    TRIMA Lookback
    """
    if optInTimePeriod == 0:
        optInTimePeriod = 30
    elif optInTimePeriod < 2 or optInTimePeriod > 100000:
        return -1

    return optInTimePeriod - 1

@cython.boundscheck(False)
@cython.wraparound(False)
def TA_TRIMA(
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
        return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
    if endIdx < 0 or endIdx < startIdx:
        return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX

    if optInTimePeriod == 0:
        optInTimePeriod = 30
    elif optInTimePeriod < 2 or optInTimePeriod > 100000:
        return TA_RetCode.TA_BAD_PARAM

    lookbackTotal = optInTimePeriod - 1

    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    outIdx = 0
    if (optInTimePeriod % 2) == 1:
        # Logic for Odd period
        i = (optInTimePeriod >> 1)
        factor = (i + 1) * (i + 1)
        factor = 1.0 / factor

        trailingIdx = startIdx - lookbackTotal
        middleIdx = trailingIdx + i
        todayIdx = middleIdx + i
        numerator = 0.0
        numeratorSub = 0.0

        for i in range(middleIdx, trailingIdx - 1, -1):
            tempReal = inReal[i]
            numeratorSub += tempReal
            numerator += numeratorSub

        numeratorAdd = 0.0
        middleIdx += 1
        for i in range(middleIdx, todayIdx + 1):
            tempReal = inReal[i]
            numeratorAdd += tempReal
            numerator += numeratorAdd

        outIdx = 0
        tempReal = inReal[trailingIdx]
        trailingIdx += 1
        outReal[outIdx] = numerator * factor
        outIdx += 1
        todayIdx += 1

        while todayIdx <= endIdx:
            numerator -= numeratorSub
            numeratorSub -= tempReal
            tempReal = inReal[middleIdx]
            middleIdx += 1
            numeratorSub += tempReal

            numerator += numeratorAdd
            numeratorAdd -= tempReal
            tempReal = inReal[todayIdx]
            todayIdx += 1
            numeratorAdd += tempReal

            numerator += tempReal

            tempReal = inReal[trailingIdx]
            trailingIdx += 1
            outReal[outIdx] = numerator * factor
            outIdx += 1
    else:
        # Even logic
        i = (optInTimePeriod >> 1)
        factor = i * (i + 1)
        factor = 1.0 / factor

        trailingIdx = startIdx - lookbackTotal
        middleIdx = trailingIdx + i - 1
        todayIdx = middleIdx + i
        numerator = 0.0
        numeratorSub = 0.0

        for i in range(middleIdx, trailingIdx - 1, -1):
            tempReal = inReal[i]
            numeratorSub += tempReal
            numerator += numeratorSub

        numeratorAdd = 0.0
        middleIdx += 1
        for i in range(middleIdx, todayIdx + 1):
            tempReal = inReal[i]
            numeratorAdd += tempReal
            numerator += numeratorAdd

        outIdx = 0
        tempReal = inReal[trailingIdx]
        trailingIdx += 1
        outReal[outIdx] = numerator * factor
        outIdx += 1
        todayIdx += 1

        while todayIdx <= endIdx:
            numerator -= numeratorSub
            numeratorSub -= tempReal
            tempReal = inReal[middleIdx]
            middleIdx += 1
            numeratorSub += tempReal

            numeratorAdd -= tempReal
            numerator += numeratorAdd
            tempReal = inReal[todayIdx]
            todayIdx += 1
            numeratorAdd += tempReal

            numerator += tempReal

            tempReal = inReal[trailingIdx]
            trailingIdx += 1
            outReal[outIdx] = numerator * factor
            outIdx += 1

    outNBElement[0] = outIdx
    outBegIdx[0] = startIdx

    return TA_RetCode.TA_SUCCESS

def TRIMA(real: np.ndarray, timeperiod: int = 30):
    """TRIMA(real, timeperiod=30)

    Triangular Moving Average

    Inputs:
        real: (any ndarray)
        timeperiod: (int) Number of period
    Outputs:
        trima: (ndarray) Triangular Moving Average
    """
    real = check_array(real)

    outReal = np.full_like(real, np.nan)
    length: cython.Py_ssize_t = real.shape[0]

    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback = startIdx + TA_TRIMA_Lookback(timeperiod)

    outBegIdx: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.int64)
    outNBElement: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.int64)

    TA_TRIMA(0, endIdx, real[startIdx:], timeperiod,
            outBegIdx, outNBElement, outReal[lookback:])
    return outReal 