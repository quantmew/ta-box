import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
from .ta_utility import TA_GLOBALS_UNSTABLE_PERIOD, TA_FuncUnstId
from ..settings import TA_FUNC_NO_RANGE_CHECK

if not cython.compiled:
    from .ta_utility import TA_INTEGER_DEFAULT


def TA_MINUS_DM_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """
    TA_MINUS_DM_Lookback - Minus Directional Movement Lookback

    Input:
        optInTimePeriod: (int) Number of period (From 1 to 100000)

    Output:
        (int) Number of lookback periods
    """
    if not TA_FUNC_NO_RANGE_CHECK:
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 14
        elif optInTimePeriod < 1 or optInTimePeriod > 100000:
            return -1

    if optInTimePeriod > 1:
        return (
            optInTimePeriod
            + TA_GLOBALS_UNSTABLE_PERIOD(TA_FuncUnstId.TA_FUNC_UNST_MINUS_DM)
            - 1
        )
    else:
        return 1


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_MINUS_DM(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inHigh: cython.double[::1],
    inLow: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """TA_MINUS_DM - Minus Directional Movement

    Input  = High, Low
    Output = double

    Optional Parameters
    -------------------
    optInTimePeriod:(From 1 to 100000)
       Number of period
    """
    # Parameter checks
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if (endIdx < 0) or (endIdx < startIdx):
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        if inHigh is None or inLow is None:
            return TA_RetCode.TA_BAD_PARAM

        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 14
        elif optInTimePeriod < 1 or optInTimePeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM

        if outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    today: cython.Py_ssize_t
    lookbackTotal: cython.Py_ssize_t
    outIdx: cython.Py_ssize_t
    prevHigh: cython.double
    prevLow: cython.double
    tempReal: cython.double
    prevMinusDM: cython.double
    diffP: cython.double  # Plus Delta
    diffM: cython.double  # Minus Delta
    i: cython.Py_ssize_t

    """
    The DM1 (one period) is base on the largest part of
    today's range that is outside of yesterdays range.
    
    The following 7 cases explain how the +DM and -DM are
    calculated on one period:
    
    Case 1:                       Case 2:
        C|                        A|
         |                         | C|
         | +DM1 = (C-A)           B|  | +DM1 = 0
         | -DM1 = 0                   | -DM1 = (B-D)
        A|  |                           D| 
         | D|                    
        B|
        
    Case 3:                       Case 4:
        C|                           C|
         |                        A|  |
         | +DM1 = (C-A)            |  | +DM1 = 0
         | -DM1 = 0               B|  | -DM1 = (B-D)
        A|  |                            | 
         |  |                           D|
        B|  |
            D|
            
    Case 5:                      Case 6:
        A|                           A| C|
         | C| +DM1 = 0                |  |  +DM1 = 0
         |  | -DM1 = 0                |  |  -DM1 = 0
         | D|                         |  |
        B|                           B| D|
        
    Case 7:
        C|
        A|  |
         |  | +DM=0
        B|  | -DM=0
            D|
            
    In case 3 and 4, the rule is that the smallest delta between
    (C-A) and (B-D) determine which of +DM or -DM is zero.
    
    In case 7, (C-A) and (B-D) are equal, so both +DM and -DM are
    zero.
    
    The rules remain the same when A=B and C=D (when the highs
    equal the lows).
    
    When calculating the DM over a period > 1, the one-period DM
    for the desired period are initialy sum. In other word, 
    for a -DM14, sum the -DM1 for the first 14 days (that's 
    13 values because there is no DM for the first day!)
    Subsequent DM are calculated using the Wilder's
    smoothing approach:
    
                                    Previous -DM14
    Today's -DM14 = Previous -DM14 -  -------------- + Today's -DM1
                                             14
                                             
    Reference:
        New Concepts In Technical Trading Systems, J. Welles Wilder Jr
    """

    if optInTimePeriod > 1:
        lookbackTotal = (
            optInTimePeriod
            + TA_GLOBALS_UNSTABLE_PERIOD(TA_FuncUnstId.TA_FUNC_UNST_MINUS_DM)
            - 1
        )
    else:
        lookbackTotal = 1

    # Adjust startIdx to account for the lookback period
    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    # Make sure there is still something to evaluate
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    # Indicate where the next output should be put in the outReal
    outIdx = 0

    # Trap the case where no smoothing is needed
    if optInTimePeriod <= 1:
        # No smoothing needed. Just do a simple DM1 for each price bar
        outBegIdx[0] = startIdx
        today = startIdx - 1
        prevHigh = inHigh[today]
        prevLow = inLow[today]
        while today < endIdx:
            today += 1
            tempReal = inHigh[today]
            diffP = tempReal - prevHigh  # Plus Delta
            prevHigh = tempReal
            tempReal = inLow[today]
            diffM = prevLow - tempReal  # Minus Delta
            prevLow = tempReal
            if (diffM > 0) and (diffP < diffM):
                # Case 2 and 4: +DM=0,-DM=diffM
                outReal[outIdx] = diffM
            else:
                outReal[outIdx] = 0.0
            outIdx += 1

        outNBElement[0] = outIdx
        return TA_RetCode.TA_SUCCESS

    # Process the initial DM
    outBegIdx[0] = startIdx
    prevMinusDM = 0.0
    today = startIdx - lookbackTotal
    prevHigh = inHigh[today]
    prevLow = inLow[today]
    i = optInTimePeriod - 1
    while i > 0:
        i -= 1
        today += 1
        tempReal = inHigh[today]
        diffP = tempReal - prevHigh  # Plus Delta
        prevHigh = tempReal
        tempReal = inLow[today]
        diffM = prevLow - tempReal  # Minus Delta
        prevLow = tempReal

        if (diffM > 0) and (diffP < diffM):
            # Case 2 and 4: +DM=0,-DM=diffM
            prevMinusDM += diffM

    # Process subsequent DM
    # Skip the unstable period
    i = TA_GLOBALS_UNSTABLE_PERIOD(TA_FuncUnstId.TA_FUNC_UNST_MINUS_DM)
    while i > 0:
        i -= 1
        today += 1
        tempReal = inHigh[today]
        diffP = tempReal - prevHigh  # Plus Delta
        prevHigh = tempReal
        tempReal = inLow[today]
        diffM = prevLow - tempReal  # Minus Delta
        prevLow = tempReal
        if (diffM > 0) and (diffP < diffM):
            # Case 2 and 4: +DM=0,-DM=diffM
            prevMinusDM = prevMinusDM - (prevMinusDM / optInTimePeriod) + diffM
        else:
            # Case 1,3,5 and 7
            prevMinusDM = prevMinusDM - (prevMinusDM / optInTimePeriod)

    # Now start to write the output in the caller provided outReal
    outReal[0] = prevMinusDM
    outIdx = 1

    while today < endIdx:
        today += 1
        tempReal = inHigh[today]
        diffP = tempReal - prevHigh  # Plus Delta
        prevHigh = tempReal
        tempReal = inLow[today]
        diffM = prevLow - tempReal  # Minus Delta
        prevLow = tempReal

        if (diffM > 0) and (diffP < diffM):
            # Case 2 and 4: +DM=0,-DM=diffM
            prevMinusDM = prevMinusDM - (prevMinusDM / optInTimePeriod) + diffM
        else:
            # Case 1,3,5 and 7
            prevMinusDM = prevMinusDM - (prevMinusDM / optInTimePeriod)

        outReal[outIdx] = prevMinusDM
        outIdx += 1

    outNBElement[0] = outIdx
    return TA_RetCode.TA_SUCCESS

def MINUS_DM(high: np.ndarray, low: np.ndarray, timeperiod: int = 14) -> np.ndarray:
    """MINUS_DM(high, low[, timeperiod=14])

    Minus Directional Movement (Overlap Studies)

    The MINUS_DM is calculated based on the largest part of today's range
    that is outside of yesterdays range.

    Inputs:
        high: (any ndarray) High prices
        low: (any ndarray) Low prices
    Parameters:
        timeperiod: 14 Number of periods
    Outputs:
        real
    """
    high = check_array(high)
    low = check_array(low)

    if high.shape != low.shape:
        raise ValueError("high and low must have the same shape")

    check_timeperiod(timeperiod)

    length: cython.Py_ssize_t = high.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(high)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_MINUS_DM_Lookback(timeperiod)

    outReal = np.full_like(high, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_MINUS_DM(
        0,
        endIdx,
        high[startIdx:],
        low[startIdx:],
        timeperiod,
        outBegIdx,
        outNBElement,
        outReal[lookback:],
    )

    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal
    return outReal
