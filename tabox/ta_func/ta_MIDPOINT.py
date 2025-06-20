import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
from ..settings import TA_FUNC_NO_RANGE_CHECK
from .ta_utility import TA_INTEGER_DEFAULT


def TA_MIDPOINT_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """
    TA_MIDPOINT_Lookback - MidPoint over period Lookback

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
def TA_MIDPOINT(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """TA_MIDPOINT - MidPoint over period

    Input  = double
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
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 14
        elif optInTimePeriod < 2 or optInTimePeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM
        if inReal is None or outReal is None:
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
        # Fix: Initialize lowest and highest with the first value in the range
        lowest = inReal[trailingIdx]
        highest = lowest
        # Loop through the range from trailingIdx to today (inclusive)
        for i in range(trailingIdx, today + 1):
            tmp = inReal[i]
            if tmp < lowest:
                lowest = tmp
            elif tmp > highest:
                highest = tmp
        outReal[outIdx] = (highest + lowest) / 2.0
        outIdx += 1
        today += 1
        trailingIdx += 1

    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx
    return TA_RetCode.TA_SUCCESS


def MIDPOINT(real: np.ndarray, timeperiod: int = 14) -> np.ndarray:
    """MIDPOINT(real[, timeperiod=14])

    MidPoint over period (Overlap Studies)

    The midpoint is calculated as (Highest Value + Lowest Value)/2 over a specified period.

    Inputs:
        real: (any ndarray) Input series
    Parameters:
        timeperiod: 14 Number of periods
    Outputs:
        real
    """
    real = check_array(real)
    check_timeperiod(timeperiod)

    length: cython.Py_ssize_t = real.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_MIDPOINT_Lookback(timeperiod)

    outReal = np.full_like(real, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_MIDPOINT(
        0,
        endIdx,
        real[startIdx:],
        timeperiod,
        outBegIdx,
        outNBElement,
        outReal[lookback:],
    )
    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal
    return outReal
