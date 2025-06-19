import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
from .ta_utility import TA_INTEGER_DEFAULT
from ..settings import TA_FUNC_NO_RANGE_CHECK

@cython.boundscheck(False)
@cython.wraparound(False)
def TA_AROON_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """
    TA_AROON_Lookback - Aroon Lookback
    
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
    return optInTimePeriod

@cython.boundscheck(False)
@cython.wraparound(False)
def TA_INT_AROON(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inHigh: cython.double[::1],
    inLow: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outAroonDown: cython.double[::1],
    outAroonUp: cython.double[::1],
) -> cython.int:
    """Internal AROON implementation without parameter checks"""
    # This function is using a speed optimized algorithm
    # for the min/max logic.
    #
    # You might want to first look at how TA_MIN/TA_MAX works
    # and this function will become easier to understand.
    
    # Move up the start index if there is not
    # enough initial data.
    if startIdx < optInTimePeriod:
        startIdx = optInTimePeriod
    
    # Make sure there is still something to evaluate.
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS
    
    # Proceed with the calculation for the requested range.
    # Note that this algorithm allows the input and
    # output to be the same buffer.
    outIdx: cython.Py_ssize_t = 0
    today: cython.Py_ssize_t = startIdx
    trailingIdx: cython.Py_ssize_t = startIdx - optInTimePeriod
    lowestIdx: cython.Py_ssize_t = -1
    highestIdx: cython.Py_ssize_t = -1
    lowest: cython.double = 0.0
    highest: cython.double = 0.0
    factor: cython.double = 100.0 / optInTimePeriod
    
    while today <= endIdx:
        # Keep track of the lowestIdx
        tmp: cython.double = inLow[today]
        if lowestIdx < trailingIdx:
            lowestIdx = trailingIdx
            lowest = inLow[lowestIdx]
            i: cython.Py_ssize_t = lowestIdx
            while i <= today:
                tmp = inLow[i]
                if tmp <= lowest:
                    lowestIdx = i
                    lowest = tmp
                i += 1
        elif tmp <= lowest:
            lowestIdx = today
            lowest = tmp
        
        # Keep track of the highestIdx
        tmp = inHigh[today]
        if highestIdx < trailingIdx:
            highestIdx = trailingIdx
            highest = inHigh[highestIdx]
            i = highestIdx
            while i <= today:
                tmp = inHigh[i]
                if tmp >= highest:
                    highestIdx = i
                    highest = tmp
                i += 1
        elif tmp >= highest:
            highestIdx = today
            highest = tmp
        
        # Note: Do not forget that input and output buffer can be the same,
        #       so writing to the output is the last thing being done here.
        outAroonUp[outIdx] = factor * (optInTimePeriod - (today - highestIdx))
        outAroonDown[outIdx] = factor * (optInTimePeriod - (today - lowestIdx))
        
        outIdx += 1
        trailingIdx += 1
        today += 1
    
    # Keep the outBegIdx relative to the
    # caller input before returning.
    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx
    return TA_RetCode.TA_SUCCESS

def TA_AROON(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inHigh: cython.double[::1],
    inLow: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outAroonDown: cython.double[::1],
    outAroonUp: cython.double[::1],
) -> cython.int:
    """TA_AROON - Aroon
    
    Input  = High, Low
    Output = double, double
    
    Optional Parameters
    -------------------
    optInTimePeriod:(From 2 to 100000)
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
        if outAroonDown is None or outAroonUp is None:
            return TA_RetCode.TA_BAD_PARAM
    return TA_INT_AROON(
        startIdx, endIdx, inHigh, inLow, optInTimePeriod,
        outBegIdx, outNBElement, outAroonDown, outAroonUp
    )

def AROON(
    high: np.ndarray,
    low: np.ndarray,
    timeperiod: int = 14
) -> tuple[np.ndarray, np.ndarray]:
    """AROON(high, low[, timeperiod=14])
    
    Aroon (Overlap Studies)
    
    The Aroon indicator consists of two lines: Aroon Up and Aroon Down.
    Aroon Up measures the strength of an uptrend, while Aroon Down measures
    the strength of a downtrend.
    
    Inputs:
        high: (any ndarray) High price series
        low: (any ndarray) Low price series
    Parameters:
        timeperiod: 14 Number of periods for the lookback
    Outputs:
        (outAroonDown, outAroonUp)
    """
    high = check_array(high)
    low = check_array(low)
    check_timeperiod(timeperiod)
    
    if high.shape[0] != low.shape[0]:
        raise ValueError("High and low arrays must have the same length")
    
    length: cython.Py_ssize_t = high.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(high)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_AROON_Lookback(timeperiod)
    
    outAroonDown = np.full_like(high, np.nan)
    outAroonUp = np.full_like(high, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)
    
    retCode = TA_AROON(
        0, endIdx, high[startIdx:], low[startIdx:], timeperiod,
        outBegIdx, outNBElement, outAroonDown[lookback:], outAroonUp[lookback:],
    )
    
    if retCode != TA_RetCode.TA_SUCCESS:
        return outAroonDown, outAroonUp
    return outAroonDown, outAroonUp
