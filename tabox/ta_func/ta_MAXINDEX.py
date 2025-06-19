import cython
import numpy as np
from .ta_utils import check_array, check_begidx1
from ..retcode import TA_RetCode
from .ta_utility import TA_INTEGER_DEFAULT
from ..settings import TA_FUNC_NO_RANGE_CHECK

def TA_MAXINDEX_Lookback(optInTimePeriod: cython.Py_ssize_t) -> cython.Py_ssize_t:
    """TA_MAXINDEX_Lookback(optInTimePeriod) -> int

    MAXINDEX Lookback

    Inputs:
        optInTimePeriod: (int) Number of period (From 2 to 100000)
    Outputs:
        int
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
def TA_MAXINDEX(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outInteger: cython.Py_ssize_t[::1]
) -> cython.int:
    """TA_MAXINDEX - Index of highest value over a specified period

    Input  = double
    Output = int

    Optional Parameters:
    -------------------
    optInTimePeriod:(From 2 to 100000)
        Number of period

    Parameters:
    -----------
    startIdx: Starting index for calculation
    endIdx: Ending index for calculation
    inReal: Input array of real numbers
    optInTimePeriod: Number of periods to look back
    outBegIdx: Output begin index
    outNBElement: Number of output elements
    outInteger: Output array of indices

    Returns:
    --------
    TA_RetCode: Return code indicating success or failure
    """
    # Validate the requested output range
    if startIdx < 0:
        return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
    if endIdx < 0 or endIdx < startIdx:
        return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
    
    # Check if input array is valid
    if inReal is None:
        return TA_RetCode.TA_BAD_PARAM
    
    # Min/max are checked for optInTimePeriod
    if optInTimePeriod == -1:  # TA_INTEGER_DEFAULT
        optInTimePeriod = 30
    elif optInTimePeriod < 2 or optInTimePeriod > 100000:
        return TA_RetCode.TA_BAD_PARAM
    
    # Check if output array is valid
    if outInteger is None:
        return TA_RetCode.TA_BAD_PARAM

    # Identify the minimum number of price bar needed
    # to identify at least one output over the specified period
    nbInitialElementNeeded: cython.int = optInTimePeriod - 1

    # Move up the start index if there is not enough initial data
    if startIdx < nbInitialElementNeeded:
        startIdx = nbInitialElementNeeded

    # Make sure there is still something to evaluate
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    # Proceed with the calculation for the requested range
    outIdx: cython.Py_ssize_t = 0
    today: cython.Py_ssize_t = startIdx
    trailingIdx: cython.Py_ssize_t = startIdx - nbInitialElementNeeded
    highestIdx: cython.Py_ssize_t = -1
    highest: cython.double = 0.0

    while today <= endIdx:
        tmp: cython.double = inReal[today]

        if highestIdx < trailingIdx:
            highestIdx = trailingIdx
            highest = inReal[highestIdx]
            i: cython.Py_ssize_t = highestIdx
            while i + 1 <= today:
                i += 1
                tmp = inReal[i]
                if tmp > highest:
                    highestIdx = i
                    highest = tmp
        elif tmp >= highest:
            highestIdx = today
            highest = tmp

        outInteger[outIdx] = highestIdx
        outIdx += 1
        trailingIdx += 1
        today += 1

    # Keep the outBegIdx relative to the caller input before returning
    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx

    return TA_RetCode.TA_SUCCESS

def MAXINDEX(real: np.ndarray, timeperiod: int = 30) -> np.ndarray:
    """MAXINDEX(real[, timeperiod=30])

    Index of highest value over a specified period

    Inputs:
        real: (any ndarray) Input array of real numbers
        timeperiod: (int) Number of period (default: 30)
    Outputs:
        real: Array of indices of highest values
    """
    real = check_array(real)
    length: cython.Py_ssize_t = real.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_MAXINDEX_Lookback(timeperiod)
    
    outInteger = np.full_like(real, 0, dtype=np.intp)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)
    
    TA_MAXINDEX(0, endIdx, real[startIdx:], timeperiod, outBegIdx, outNBElement, outInteger[lookback:])
    return outInteger