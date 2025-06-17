import cython
import numpy as np
from .ta_utils import check_array, check_begidx1
from ..retcode import *

def TA_KAMA_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """TA_KAMA_Lookback(optInTimePeriod) -> Py_ssize_t

    KAMA Lookback
    """
    if optInTimePeriod == 0:
        optInTimePeriod = 30
    elif optInTimePeriod < 2 or optInTimePeriod > 100000:
        return -1

    return optInTimePeriod + 2  # TA_GLOBALS_UNSTABLE_PERIOD(TA_FUNC_UNST_KAMA,Kama)

@cython.boundscheck(False)
@cython.wraparound(False)
def TA_KAMA(
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

    constMax = 2.0 / (30.0 + 1.0)
    constDiff = 2.0 / (2.0 + 1.0) - constMax

    lookbackTotal = optInTimePeriod + 2  # TA_GLOBALS_UNSTABLE_PERIOD(TA_FUNC_UNST_KAMA,Kama)

    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_SUCCESS

    sumROC1 = 0.0
    today = startIdx - lookbackTotal
    trailingIdx = today
    i = optInTimePeriod
    while i > 0:
        tempReal = inReal[today]
        today += 1
        tempReal -= inReal[today]
        sumROC1 += abs(tempReal)
        i -= 1

    prevKAMA = inReal[today - 1]

    tempReal = inReal[today]
    tempReal2 = inReal[trailingIdx]
    trailingIdx += 1
    periodROC = tempReal - tempReal2

    trailingValue = tempReal2

    if sumROC1 <= periodROC or sumROC1 == 0:
        tempReal = 1.0
    else:
        tempReal = abs(periodROC / sumROC1)

    tempReal = (tempReal * constDiff) + constMax
    tempReal *= tempReal

    prevKAMA = ((inReal[today] - prevKAMA) * tempReal) + prevKAMA
    today += 1

    while today <= startIdx:
        tempReal = inReal[today]
        tempReal2 = inReal[trailingIdx]
        trailingIdx += 1
        periodROC = tempReal - tempReal2

        sumROC1 -= abs(trailingValue - tempReal2)
        sumROC1 += abs(tempReal - inReal[today - 1])

        trailingValue = tempReal2

        if sumROC1 <= periodROC or sumROC1 == 0:
            tempReal = 1.0
        else:
            tempReal = abs(periodROC / sumROC1)

        tempReal = (tempReal * constDiff) + constMax
        tempReal *= tempReal

        prevKAMA = ((inReal[today] - prevKAMA) * tempReal) + prevKAMA
        today += 1

    outReal[0] = prevKAMA
    outIdx = 1
    outBegIdx[0] = today - 1

    while today <= endIdx:
        tempReal = inReal[today]
        tempReal2 = inReal[trailingIdx]
        trailingIdx += 1
        periodROC = tempReal - tempReal2

        sumROC1 -= abs(trailingValue - tempReal2)
        sumROC1 += abs(tempReal - inReal[today - 1])

        trailingValue = tempReal2

        if sumROC1 <= periodROC or sumROC1 == 0:
            tempReal = 1.0
        else:
            tempReal = abs(periodROC / sumROC1)

        tempReal = (tempReal * constDiff) + constMax
        tempReal *= tempReal

        prevKAMA = ((inReal[today] - prevKAMA) * tempReal) + prevKAMA
        today += 1
        outReal[outIdx] = prevKAMA
        outIdx += 1

    outNBElement[0] = outIdx

    return TA_SUCCESS

def KAMA(real: np.ndarray, timeperiod: int = 30):
    """KAMA(real, timeperiod=30)

    Kaufman Adaptive Moving Average

    Inputs:
        real: (any ndarray)
        timeperiod: (int) Number of period
    Outputs:
        kama: (ndarray) Kaufman Adaptive Moving Average
    """
    real = check_array(real)

    outReal = np.full_like(real, np.nan)
    length: cython.Py_ssize_t = real.shape[0]

    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback = startIdx + TA_KAMA_Lookback(timeperiod)

    outBegIdx: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.int64)
    outNBElement: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.int64)

    TA_KAMA(0, endIdx, real[startIdx:], timeperiod,
            outBegIdx, outNBElement, outReal[lookback:])
    return outReal
