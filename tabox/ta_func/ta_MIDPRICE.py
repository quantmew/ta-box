import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
from ..settings import TA_FUNC_NO_RANGE_CHECK
if not cython.compiled:
    from .ta_utility import TA_INTEGER_DEFAULT


def TA_MIDPRICE_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """
    TA_MIDPRICE_Lookback - Midpoint Price over period Lookback

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
def TA_MIDPRICE(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inHigh: cython.double[::1],
    inLow: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """TA_MIDPRICE - Midpoint Price over period

    Input  = High, Low
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
        if inHigh is None or inLow is None:
            return TA_RetCode.TA_BAD_PARAM
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 14
        elif optInTimePeriod < 2 or optInTimePeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM
        if outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    lowest: cython.double
    highest: cython.double
    tmp: cython.double
    outIdx: cython.Py_ssize_t
    nbInitialElementNeeded: cython.Py_ssize_t
    trailingIdx: cython.Py_ssize_t
    today: cython.Py_ssize_t
    i: cython.Py_ssize_t

    # Identify the minimum number of price bar needed to identify at least one output over the specified period.
    nbInitialElementNeeded = optInTimePeriod - 1

    # Move up the start index if there is not enough initial data.
    if startIdx < nbInitialElementNeeded:
        startIdx = nbInitialElementNeeded

    # Make sure there is still something to evaluate.
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    # Proceed with the calculation for the requested range.
    outIdx = 0
    today = startIdx
    trailingIdx = startIdx - nbInitialElementNeeded

    while today <= endIdx:
        lowest = inLow[trailingIdx]
        highest = inHigh[trailingIdx]
        trailingIdx += 1
        for i in range(trailingIdx, today + 1):
            tmp = inLow[i]
            if tmp < lowest:
                lowest = tmp
            tmp = inHigh[i]
            if tmp > highest:
                highest = tmp

        outReal[outIdx] = (highest + lowest) / 2.0
        outIdx += 1
        today += 1

    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx
    return TA_RetCode.TA_SUCCESS


def MIDPRICE(
    realHigh: np.ndarray, realLow: np.ndarray, timeperiod: int = 14
) -> np.ndarray:
    """MIDPRICE(realHigh, realLow[, timeperiod=14])

    Midpoint Price over period (Overlap Studies)

    MIDPRICE = (Highest High + Lowest Low) / 2

    This function is equivalent to MEDPRICE when the period is 1.

    Inputs:
        realHigh: (any ndarray) High prices
        realLow: (any ndarray) Low prices
    Parameters:
        timeperiod: 14 Number of period
    Outputs:
        real
    """
    realHigh = check_array(realHigh)
    realLow = check_array(realLow)
    check_timeperiod(timeperiod)

    if realHigh.shape[0] != realLow.shape[0]:
        raise ValueError("High and low arrays must have the same length")

    length: cython.Py_ssize_t = realHigh.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(realHigh)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_MIDPRICE_Lookback(timeperiod)

    outReal = np.full_like(realHigh, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_MIDPRICE(
        0,
        endIdx,
        realHigh[startIdx:],
        realLow[startIdx:],
        timeperiod,
        outBegIdx,
        outNBElement,
        outReal[lookback:],
    )
    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal
    return outReal
