import cython
import numpy as np
from .ta_utils import check_array, check_begidx1
from ..retcode import TA_RetCode
from .ta_MINUS_DM import TA_MINUS_DM
from ..settings import TA_FUNC_NO_RANGE_CHECK
from .ta_defs import TA_REAL_DEFAULT

@cython.boundscheck(False)
@cython.wraparound(False)
def TA_SAR_Lookback(optInAcceleration: cython.double, optInMaximum: cython.double) -> cython.Py_ssize_t:
    """
    TA_SAR_Lookback - Parabolic SAR Lookback
    
    Input:
        optInAcceleration: (double) Acceleration Factor used up to the Maximum value (From 0 to TA_REAL_MAX)
        optInMaximum: (double) Acceleration Factor Maximum value (From 0 to TA_REAL_MAX)
    
    Output:
        (int) Number of lookback periods
    """
    if not TA_FUNC_NO_RANGE_CHECK:
        if optInAcceleration == TA_REAL_DEFAULT:  # TA_REAL_DEFAULT is treated as 0.0 here
            optInAcceleration = 0.02
        elif optInAcceleration < 0.0 or optInAcceleration > 3.0e37:
            return -1
            
        if optInMaximum == TA_REAL_DEFAULT:  # TA_REAL_DEFAULT is treated as 0.0 here
            optInMaximum = 0.2
        elif optInMaximum < 0.0 or optInMaximum > 3.0e37:
            return -1
    
    # SAR always sacrifices one price bar to establish the initial extreme price
    return 1

@cython.boundscheck(False)
@cython.wraparound(False)
def TA_SAR(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inHigh: cython.double[::1],
    inLow: cython.double[::1],
    optInAcceleration: cython.double,
    optInMaximum: cython.double,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1]
) -> cython.int:
    """
    TA_SAR - Parabolic SAR
    
    Input  = High, Low
    Output = double
    
    Optional Parameters
    -------------------
    optInAcceleration:(From 0 to TA_REAL_MAX)
       Acceleration Factor used up to the Maximum value
    optInMaximum:(From 0 to TA_REAL_MAX)
       Acceleration Factor Maximum value
    """
    # Parameter checks
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        if inHigh is None or inLow is None:
            return TA_RetCode.TA_BAD_PARAM
        if outReal is None:
            return TA_RetCode.TA_BAD_PARAM
            
        if optInAcceleration == TA_REAL_DEFAULT:  # TA_REAL_DEFAULT
            optInAcceleration = 0.02
        elif optInAcceleration < 0.0 or optInAcceleration > 3.0e37:
            return TA_RetCode.TA_BAD_PARAM
            
        if optInMaximum == TA_REAL_DEFAULT:  # TA_REAL_DEFAULT
            optInMaximum = 0.2
        elif optInMaximum < 0.0 or optInMaximum > 3.0e37:
            return TA_RetCode.TA_BAD_PARAM
    
    # Local variables
    isLong: cython.int  # > 0 indicates long, == 0 indicates short
    todayIdx: cython.Py_ssize_t
    outIdx: cython.Py_ssize_t
    retCode: cython.int
    tempInt1: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    tempInt2: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    ep_temp: cython.double[::1] = np.zeros(1, dtype=np.double)
    
    newHigh: cython.double
    newLow: cython.double
    prevHigh: cython.double
    prevLow: cython.double
    af: cython.double
    ep: cython.double
    sar: cython.double
    
    # Ensure startIdx is at least 1
    if startIdx < 1:
        startIdx = 1
    
    # Check if there's data to process
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS
    
    # Correct acceleration factor if it exceeds maximum
    af = optInAcceleration
    if af > optInMaximum:
        af = optInAcceleration = optInMaximum
    
    # Determine initial direction (long or short)
    # by comparing +DM and -DM between first and second bar
    retCode = TA_MINUS_DM(
        startIdx, startIdx, inHigh, inLow, 1,
        tempInt1, tempInt2, ep_temp
    )
    
    if retCode == TA_RetCode.TA_SUCCESS:
        if ep_temp[0] > 0:
            isLong = 0  # short
        else:
            isLong = 1  # long
    else:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return retCode
    
    outBegIdx[0] = startIdx
    outIdx = 0
    
    # Calculate first SAR value
    todayIdx = startIdx
    newHigh = inHigh[todayIdx - 1]
    newLow = inLow[todayIdx - 1]
    
    # SAR_ROUNDING is not implemented as per comment in C code
    # #define SAR_ROUNDING(x) x=round_pos_2(x)
    # We keep it as is for now
    
    if isLong == 1:
        ep = inHigh[todayIdx]
        sar = newLow
    else:
        ep = inLow[todayIdx]
        sar = newHigh
    
    # Cheat on newLow and newHigh for the first iteration
    newLow = inLow[todayIdx]
    newHigh = inHigh[todayIdx]

    # Main calculation loop
    while todayIdx <= endIdx:
        prevLow = newLow
        prevHigh = newHigh
        newLow = inLow[todayIdx]
        newHigh = inHigh[todayIdx]
        todayIdx += 1
        
        if isLong == 1:
            # Current direction is long
            if newLow <= sar:
                # Switch to short
                isLong = 0
                sar = ep
                
                # Ensure SAR is within yesterday's and today's range
                if sar < prevHigh:
                    sar = prevHigh
                if sar < newHigh:
                    sar = newHigh
                
                # Output the overridden SAR
                outReal[outIdx] = sar
                outIdx += 1
                
                # Adjust acceleration factor and extreme point
                af = optInAcceleration
                ep = newLow
                
                # Calculate new SAR
                sar = sar + af * (ep - sar)
                
                # Ensure new SAR is within range
                if sar < prevHigh:
                    sar = prevHigh
                if sar < newHigh:
                    sar = newHigh
            else:
                # No switch, continue long
                outReal[outIdx] = sar
                outIdx += 1
                
                # Adjust extreme point and acceleration factor
                if newHigh > ep:
                    ep = newHigh
                    af += optInAcceleration
                    if af > optInMaximum:
                        af = optInMaximum
                
                # Calculate new SAR
                sar = sar + af * (ep - sar)
                
                # Ensure new SAR is within range
                if sar > prevLow:
                    sar = prevLow
                if sar > newLow:
                    sar = newLow
        else:
            # Current direction is short
            if newHigh >= sar:
                # Switch to long
                isLong = 1
                sar = ep
                
                # Ensure SAR is within yesterday's and today's range
                if sar > prevLow:
                    sar = prevLow
                if sar > newLow:
                    sar = newLow
                
                # Output the overridden SAR
                outReal[outIdx] = sar
                outIdx += 1
                
                # Adjust acceleration factor and extreme point
                af = optInAcceleration
                ep = newHigh
                
                # Calculate new SAR
                sar = sar + af * (ep - sar)
                
                # Ensure new SAR is within range
                if sar > prevLow:
                    sar = prevLow
                if sar > newLow:
                    sar = newLow
            else:
                # No switch, continue short
                outReal[outIdx] = sar
                outIdx += 1
                
                # Adjust extreme point and acceleration factor
                if newLow < ep:
                    ep = newLow
                    af += optInAcceleration
                    if af > optInMaximum:
                        af = optInMaximum
                
                # Calculate new SAR
                sar = sar + af * (ep - sar)
                
                # Ensure new SAR is within range
                if sar < prevHigh:
                    sar = prevHigh
                if sar < newHigh:
                    sar = newHigh
    
    outNBElement[0] = outIdx
    return TA_RetCode.TA_SUCCESS

def SAR(
    high: np.ndarray,
    low: np.ndarray,
    optInAcceleration: float = 0.02,
    optInMaximum: float = 0.2
) -> np.ndarray:
    """
    SAR(high, low[, optInAcceleration=0.02, optInMaximum=0.2])
    
    Parabolic Stop and Reverse (Overlap Studies)
    
    The SAR is a trend-following indicator that helps identify potential 
    trend reversals and provides stop-loss levels.
    
    Inputs:
        high: (ndarray) High prices
        low: (ndarray) Low prices
    Parameters:
        optInAcceleration: 0.02 Acceleration factor
        optInMaximum: 0.2 Maximum acceleration factor
    Outputs:
        real: Parabolic SAR values
    """
    high = check_array(high)
    low = check_array(low)
    
    if high.shape[0] != low.shape[0]:
        raise ValueError("High and low arrays must have the same length")
    
    length: cython.Py_ssize_t = high.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(high)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_SAR_Lookback(optInAcceleration, optInMaximum)
    
    outReal = np.full_like(high, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)
    
    retCode = TA_SAR(
        0,
        endIdx,
        high[startIdx:],
        low[startIdx:],
        optInAcceleration,
        optInMaximum,
        outBegIdx,
        outNBElement,
        outReal[lookback:]
    )
    
    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal
    return outReal