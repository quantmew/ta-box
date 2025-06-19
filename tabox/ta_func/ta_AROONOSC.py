import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode, TA_INTEGER_DEFAULT
from .ta_utility import TA_GLOBALS_UNSTABLE_PERIOD, TA_FuncUnstId, TA_IS_ZERO
from ..settings import TA_FUNC_NO_RANGE_CHECK

def TA_AROONOSC_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """
    TA_AROONOSC_Lookback - Aroon Oscillator Lookback
    
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
def TA_INT_AROONOSC(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inHigh: cython.double[::1],
    inLow: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """Internal AROONOSC implementation without parameter checks"""
    # This code is almost identical to the TA_AROON function
    # except that instead of outputing ArroonUp and AroonDown
    # individually, an oscillator is build from both.
    #
    #  AroonOsc = AroonUp- AroonDown;
    #
    # This function is using a speed optimized algorithm
    # for the min/max logic.
    #
    # You might want to first look at how TA_MIN/TA_MAX works
    # and this function will become easier to understand.
    
    # Initialize variables
    lowest: cython.double = 0.0
    highest: cython.double = 0.0
    tmp: cython.double = 0.0
    factor: cython.double = 0.0
    aroon: cython.double = 0.0
    outIdx: cython.int = 0
    trailingIdx: cython.int = 0
    lowestIdx: cython.int = 0
    highestIdx: cython.int = 0
    today: cython.int = 0
    i: cython.int = 0

    # Ensure there is enough initial data
    if startIdx < optInTimePeriod:
        startIdx = optInTimePeriod
    
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS
    
    # Proceed with the calculation for the requested range
    outIdx = 0
    today = startIdx
    trailingIdx = startIdx - optInTimePeriod
    lowestIdx = -1
    highestIdx = -1
    lowest = 0.0
    highest = 0.0
    factor = 100.0 / optInTimePeriod
    
    while today <= endIdx:
        # Keep track of the lowestIdx
        tmp = inLow[today]
        if lowestIdx < trailingIdx:
            lowestIdx = trailingIdx
            lowest = inLow[lowestIdx]
            i = lowestIdx
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
        
        # The oscillator calculation
        # Aroon = factor*(highestIdx-lowestIdx)
        aroon = factor * (highestIdx - lowestIdx)
        
        # Write to output
        outReal[outIdx] = aroon
        outIdx += 1
        trailingIdx += 1
        today += 1
    
    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx
    return TA_RetCode.TA_SUCCESS

def TA_AROONOSC(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inHigh: cython.double[::1],
    inLow: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """TA_AROONOSC - Aroon Oscillator
    
    Input  = High, Low
    Output = double
    
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
        if outReal is None:
            return TA_RetCode.TA_BAD_PARAM
    return TA_INT_AROONOSC(
        startIdx, endIdx, inHigh, inLow, optInTimePeriod,
        outBegIdx, outNBElement, outReal
    )

def AROONOSC(
    high: np.ndarray,
    low: np.ndarray,
    timeperiod: int = 14
) -> np.ndarray:
    """AROONOSC(high, low[, timeperiod=14])
    
    Aroon Oscillator (Overlap Studies)
    
    The Aroon Oscillator is calculated as the difference between AroonUp and AroonDown.
    AroonUp measures the strength of an uptrend, while AroonDown measures the strength of a downtrend.
    
    Inputs:
        high: (any ndarray) High price series
        low: (any ndarray) Low price series
    Parameters:
        timeperiod: 14 Number of periods
    Outputs:
        real: Aroon Oscillator values
    """
    high = check_array(high)
    low = check_array(low)
    check_timeperiod(timeperiod)
    
    if high.shape != low.shape:
        raise ValueError("High and low arrays must have the same shape")
    
    length: cython.Py_ssize_t = high.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(high)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_AROONOSC_Lookback(timeperiod)
    
    outReal = np.full_like(high, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)
    
    retCode = TA_AROONOSC(
        0, endIdx, high[startIdx:], low[startIdx:], timeperiod,
        outBegIdx, outNBElement, outReal[lookback:],
    )
    
    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal
    return outReal