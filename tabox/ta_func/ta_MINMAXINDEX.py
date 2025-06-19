import cython
import numpy as np
from .ta_utils import check_array, check_begidx1, check_timeperiod
from ..retcode import TA_RetCode
from .ta_utility import TA_INTEGER_DEFAULT
from ..settings import TA_FUNC_NO_RANGE_CHECK

def TA_MINMAXINDEX_Lookback(optInTimePeriod: cython.Py_ssize_t) -> cython.Py_ssize_t:
    """TA_MINMAXINDEX_Lookback - Lookback for Indexes of lowest and highest values over a specified period

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
def TA_MINMAXINDEX(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outMinIdx: cython.Py_ssize_t[::1],
    outMaxIdx: cython.Py_ssize_t[::1]
) -> cython.int:
    """TA_MINMAXINDEX - Indexes of lowest and highest values over a specified period

    Input  = double
    Output = int, int

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
    outMinIdx: Output array of minimum value indices
    outMaxIdx: Output array of maximum value indices

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
    
    # Check if output arrays are valid
    if outMinIdx is None or outMaxIdx is None:
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
    lowestIdx: cython.Py_ssize_t = -1
    lowest: cython.double = 0.0

    while today <= endIdx:
        tmpHigh = tmpLow = inReal[today]

        if highestIdx < trailingIdx:
            highestIdx = trailingIdx
            highest = inReal[highestIdx]
            i: cython.Py_ssize_t = highestIdx
            while i + 1 <= today:
                i += 1
                tmpHigh = inReal[i]
                if tmpHigh > highest:
                    highestIdx = i
                    highest = tmpHigh
        elif tmpHigh >= highest:
            highestIdx = today
            highest = tmpHigh

        if lowestIdx < trailingIdx:
            lowestIdx = trailingIdx
            lowest = inReal[lowestIdx]
            i = lowestIdx
            while i + 1 <= today:
                i += 1
                tmpLow = inReal[i]
                if tmpLow < lowest:
                    lowestIdx = i
                    lowest = tmpLow
        elif tmpLow <= lowest:
            lowestIdx = today
            lowest = tmpLow

        outMaxIdx[outIdx] = highestIdx
        outMinIdx[outIdx] = lowestIdx
        outIdx += 1
        trailingIdx += 1
        today += 1

    # Keep the outBegIdx relative to the caller input before returning
    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx

    return TA_RetCode.TA_SUCCESS

def MINMAXINDEX(real: np.ndarray, timeperiod: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """MINMAXINDEX(real[, timeperiod=30])

    Indexes of lowest and highest values over a specified period

    Inputs:
        real: (any ndarray) Input array of real numbers
        timeperiod: (int) Number of period (default: 30)
    Outputs:
        minidx: Array of indices of lowest values
        maxidx: Array of indices of highest values
    """
    real = check_array(real)
    check_timeperiod(timeperiod)
    
    length: cython.Py_ssize_t = real.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_MINMAXINDEX_Lookback(timeperiod)
    
    outMinIdx = np.full_like(real, 0, dtype=np.intp)
    outMaxIdx = np.full_like(real, 0, dtype=np.intp)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)
    
    TA_MINMAXINDEX(0, endIdx, real[startIdx:], timeperiod, outBegIdx, outNBElement, outMinIdx[lookback:], outMaxIdx[lookback:])
    return outMinIdx, outMaxIdx