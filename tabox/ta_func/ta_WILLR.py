import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
if not cython.compiled:
    from .ta_utility import TA_INTEGER_DEFAULT
from ..settings import TA_FUNC_NO_RANGE_CHECK


def TA_WILLR_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """
    TA_WILLR_Lookback - Williams' %R Lookback

    Input:
        optInTimePeriod: (int) Number of period (From 2 to 100000)

    Output:
        (int) Number of lookback periods
    """
    if not TA_FUNC_NO_RANGE_CHECK:
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 14
        elif optInTimePeriod < 2 or optInTimePeriod > 100000:
            return -1
    return optInTimePeriod - 1


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_WILLR(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inHigh: cython.double[::1],
    inLow: cython.double[::1],
    inClose: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """
    TA_WILLR - Williams' %R

    Input  = High, Low, Close (double)
    Output = double

    Optional Parameters
    -------------------
    optInTimePeriod: (From 2 to 100000)
        Number of period
    """
    # Parameter checks
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        if inHigh is None or inLow is None or inClose is None:
            return TA_RetCode.TA_BAD_PARAM
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 14
        elif optInTimePeriod < 2 or optInTimePeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM
        if outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    # Identify the minimum number of price bar needed
    # to identify at least one output over the specified period.
    nbInitialElementNeeded: cython.Py_ssize_t = optInTimePeriod - 1

    # Move up the start index if there is not enough initial data.
    if startIdx < nbInitialElementNeeded:
        startIdx = nbInitialElementNeeded

    # Make sure there is still something to evaluate.
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    # Proceed with the calculation for the requested range.
    outIdx: cython.Py_ssize_t = 0
    today: cython.Py_ssize_t = startIdx
    trailingIdx: cython.Py_ssize_t = startIdx - nbInitialElementNeeded
    lowestIdx: cython.Py_ssize_t = -1
    highestIdx: cython.Py_ssize_t = -1
    lowest: cython.double = 0.0
    highest: cython.double = 0.0
    diff: cython.double = 0.0
    tmp: cython.double

    while today <= endIdx:
        # Set the lowest low
        tmp = inLow[today]
        if lowestIdx < trailingIdx:
            lowestIdx = trailingIdx
            lowest = inLow[lowestIdx]
            i: cython.Py_ssize_t = lowestIdx
            while i <= today:
                tmp = inLow[i]
                if tmp < lowest:
                    lowestIdx = i
                    lowest = tmp
                i += 1
            diff = (highest - lowest) / (-100.0)
        elif tmp <= lowest:
            lowestIdx = today
            lowest = tmp
            diff = (highest - lowest) / (-100.0)

        # Set the highest high
        tmp = inHigh[today]
        if highestIdx < trailingIdx:
            highestIdx = trailingIdx
            highest = inHigh[highestIdx]
            i: cython.Py_ssize_t = highestIdx
            while i <= today:
                tmp = inHigh[i]
                if tmp > highest:
                    highestIdx = i
                    highest = tmp
                i += 1
            diff = (highest - lowest) / (-100.0)
        elif tmp >= highest:
            highestIdx = today
            highest = tmp
            diff = (highest - lowest) / (-100.0)

        if diff != 0.0:
            outReal[outIdx] = (highest - inClose[today]) / diff
        else:
            outReal[outIdx] = 0.0
        outIdx += 1

        trailingIdx += 1
        today += 1

    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx
    return TA_RetCode.TA_SUCCESS


def WILLR(
    inHigh: np.ndarray,
    inLow: np.ndarray,
    inClose: np.ndarray,
    timeperiod: int = 14,
) -> np.ndarray:
    """
    WILLR(inHigh, inLow, inClose[, timeperiod=14])

    Williams' %R (Momentum Indicators)

    The Williams' %R is a momentum indicator that measures overbought and oversold conditions.
    It is calculated as: %R = 100 * (Highest High - Close) / (Highest High - Lowest Low) * -100

    Inputs:
        inHigh: (any ndarray) High prices
        inLow: (any ndarray) Low prices
        inClose: (any ndarray) Close prices
    Parameters:
        timeperiod: 14 Number of periods
    Outputs:
        real
    """
    inHigh = check_array(inHigh)
    inLow = check_array(inLow)
    inClose = check_array(inClose)
    check_timeperiod(timeperiod)

    length: cython.Py_ssize_t = inHigh.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(inHigh)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_WILLR_Lookback(timeperiod)

    outReal = np.full_like(inHigh, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_WILLR(
        0,
        endIdx,
        inHigh[startIdx:],
        inLow[startIdx:],
        inClose[startIdx:],
        timeperiod,
        outBegIdx,
        outNBElement,
        outReal[lookback:],
    )
    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal
    return outReal
