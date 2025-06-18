import cython
import numpy as np
from .ta_utils import check_array, check_begidx1
from ..retcode import TA_RetCode, TA_INTEGER_DEFAULT
from ..settings import TA_FUNC_NO_RANGE_CHECK

def TA_MININDEX_Lookback(optInTimePeriod: cython.Py_ssize_t) -> cython.Py_ssize_t:
    """
    TA_MININDEX_Lookback - Index of lowest value over a specified period lookback
    
    Input:
        optInTimePeriod: (int) Number of period (From 2 to 100000)
    
    Output:
        (int) Number of lookback periods
    """
    if not TA_FUNC_NO_RANGE_CHECK:
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 30
        elif optInTimePeriod < 2 or optInTimePeriod > 100000:
            return -1
    return optInTimePeriod - 1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def TA_INT_MININDEX(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outInteger: cython.Py_ssize_t[::1]
) -> cython.int:
    """Internal MININDEX implementation without parameter checks"""
    lowest: cython.double
    tmp: cython.double
    outIdx: cython.Py_ssize_t
    nbInitialElementNeeded: cython.int
    trailingIdx: cython.Py_ssize_t
    lowestIdx: cython.Py_ssize_t
    today: cython.Py_ssize_t
    i: cython.Py_ssize_t

    # Identify the minimum number of price bar needed
    # to identify at least one output over the specified period.
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
    lowestIdx = -1
    lowest = 0.0

    while today <= endIdx:
        tmp = inReal[today]

        if lowestIdx < trailingIdx:
            lowestIdx = trailingIdx
            lowest = inReal[lowestIdx]
            i = lowestIdx
            while i + 1 <= today:
                i += 1
                tmp = inReal[i]
                if tmp < lowest:
                    lowestIdx = i
                    lowest = tmp
        elif tmp <= lowest:
            lowestIdx = today
            lowest = tmp

        outInteger[outIdx] = lowestIdx
        outIdx += 1
        trailingIdx += 1
        today += 1

    # Keep the outBegIdx relative to the caller input before returning
    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx

    return TA_RetCode.TA_SUCCESS

def TA_MININDEX(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outInteger: cython.Py_ssize_t[::1],
) -> cython.int:
    """TA_MININDEX - Index of lowest value over a specified period
    
    Input  = double
    Output = int
    
    Optional Parameters
    -------------------
    optInTimePeriod:(From 2 to 100000)
        Number of period
    """
    # parameters check
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        if inReal is None:
            return TA_RetCode.TA_BAD_PARAM
        # min/max are checked for optInTimePeriod
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 30
        elif optInTimePeriod < 2 or optInTimePeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM
        if outInteger is None:
            return TA_RetCode.TA_BAD_PARAM

    # Call internal implementation
    return TA_INT_MININDEX(
        startIdx, endIdx, inReal, optInTimePeriod, outBegIdx, outNBElement, outInteger
    )

def MININDEX(real: np.ndarray, timeperiod: int = 30) -> np.ndarray:
    """MININDEX(real[, timeperiod=30])
    
    Index of lowest value over a specified period
    
    Inputs:
        real: (any ndarray) Input array of real numbers
        timeperiod: (int) Number of period (default: 30)
    Outputs:
        real: Array of indices of lowest values
    """
    real = check_array(real)
    length: cython.Py_ssize_t = real.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_MININDEX_Lookback(timeperiod)
    
    outInteger = np.full_like(real, 0, dtype=np.int64)
    outBegIdx = np.zeros(1, dtype=np.int64)
    outNBElement = np.zeros(1, dtype=np.int64)
    
    TA_MININDEX(0, endIdx, real[startIdx:], timeperiod, outBegIdx, outNBElement, outInteger[lookback:])
    return outInteger