import cython
import numpy as np
from .ta_utils import check_array, check_begidx1
from ..retcode import TA_RetCode
from ..settings import TA_FUNC_NO_RANGE_CHECK
from .ta_utility import TA_GLOBALS_UNSTABLE_PERIOD, TA_FuncUnstId, TA_IS_ZERO, TA_INTEGER_DEFAULT

if not cython.compiled:
    from math import fabs


@cython.cdivision(True)
def TA_KAMA_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """TA_KAMA_Lookback(optInTimePeriod) -> Py_ssize_t

    KAMA Lookback
    """
    period: cython.int = optInTimePeriod

    # Range check
    if not TA_FUNC_NO_RANGE_CHECK:
        if period == TA_INTEGER_DEFAULT:
            period = 30
        elif period < 2 or period > 100000:
            return -1

    return period + TA_GLOBALS_UNSTABLE_PERIOD(TA_FuncUnstId.TA_FUNC_UNST_KAMA)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def TA_KAMA(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    # Local variables
    constMax: cython.double = 2.0 / (30.0 + 1.0)
    constDiff: cython.double = 2.0 / (2.0 + 1.0) - constMax
    tempReal: cython.double
    tempReal2: cython.double
    sumROC1: cython.double
    periodROC: cython.double
    prevKAMA: cython.double
    i: cython.int
    today: cython.int
    outIdx: cython.int
    lookbackTotal: cython.int
    trailingIdx: cython.int
    trailingValue: cython.double
    period: cython.int = optInTimePeriod

    # Parameter validation
    if not TA_FUNC_NO_RANGE_CHECK:
        # Validate start and end indices
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX

        # Validate time period parameter
        if period == TA_INTEGER_DEFAULT:
            period = 30
        elif period < 2 or period > 100000:
            return TA_RetCode.TA_BAD_PARAM

    # Set default return value
    outBegIdx[0] = 0
    outNBElement[0] = 0

    # Calculate the minimum required data
    lookbackTotal = period + TA_GLOBALS_UNSTABLE_PERIOD(TA_FuncUnstId.TA_FUNC_UNST_KAMA)

    # Adjust the start index to ensure there is enough historical data
    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    # Check if there is data to process
    if startIdx > endIdx:
        return TA_RetCode.TA_SUCCESS

    # Initialize variables and process lookback period
    sumROC1 = 0.0
    today = startIdx - lookbackTotal
    trailingIdx = today
    i = period

    # Calculate the initial price change total
    while i > 0:
        tempReal = inReal[today]
        today += 1
        tempReal -= inReal[today]
        sumROC1 += fabs(tempReal)
        i -= 1

    # Calculate the first KAMA value
    prevKAMA = inReal[today - 1]

    tempReal = inReal[today]
    tempReal2 = inReal[trailingIdx]
    trailingIdx += 1
    periodROC = tempReal - tempReal2

    trailingValue = tempReal2

    # Calculate the efficiency ratio
    if sumROC1 <= fabs(periodROC) or TA_IS_ZERO(sumROC1):
        tempReal = 1.0
    else:
        tempReal = fabs(periodROC / sumROC1)

    # Calculate the smoothing constant
    tempReal = (tempReal * constDiff) + constMax
    tempReal *= tempReal

    # Calculate the first KAMA value
    prevKAMA = ((inReal[today] - prevKAMA) * tempReal) + prevKAMA
    today += 1

    # Process unstable period
    while today <= startIdx:
        tempReal = inReal[today]
        tempReal2 = inReal[trailingIdx]
        trailingIdx += 1
        periodROC = tempReal - tempReal2

        # Adjust the price change total
        sumROC1 -= fabs(trailingValue - tempReal2)
        sumROC1 += fabs(tempReal - inReal[today - 1])

        trailingValue = tempReal2

        # Calculate the efficiency ratio
        if sumROC1 <= fabs(periodROC) or TA_IS_ZERO(sumROC1):
            tempReal = 1.0
        else:
            tempReal = fabs(periodROC / sumROC1)

        # Calculate the smoothing constant
        tempReal = (tempReal * constDiff) + constMax
        tempReal *= tempReal

        # Calculate the KAMA value
        prevKAMA = ((inReal[today] - prevKAMA) * tempReal) + prevKAMA
        today += 1

    # Write the first output value
    outReal[0] = prevKAMA
    outIdx = 1
    outBegIdx[0] = today - 1

    # Calculate the remaining KAMA values
    while today <= endIdx:
        tempReal = inReal[today]
        tempReal2 = inReal[trailingIdx]
        trailingIdx += 1
        periodROC = tempReal - tempReal2

        # Adjust the price change total
        sumROC1 -= fabs(trailingValue - tempReal2)
        sumROC1 += fabs(tempReal - inReal[today - 1])

        trailingValue = tempReal2

        # Calculate the efficiency ratio
        if sumROC1 <= fabs(periodROC) or TA_IS_ZERO(sumROC1):
            tempReal = 1.0
        else:
            tempReal = fabs(periodROC / sumROC1)

        # Calculate the smoothing constant
        tempReal = (tempReal * constDiff) + constMax
        tempReal *= tempReal

        # Calculate the KAMA value
        prevKAMA = ((inReal[today] - prevKAMA) * tempReal) + prevKAMA
        today += 1
        outReal[outIdx] = prevKAMA
        outIdx += 1

    # Set the output element count
    outNBElement[0] = outIdx

    return TA_RetCode.TA_SUCCESS


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

    outBegIdx: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    outNBElement: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)

    TA_KAMA(
        0,
        endIdx,
        real[startIdx:],
        timeperiod,
        outBegIdx,
        outNBElement,
        outReal[lookback:],
    )
    return outReal
