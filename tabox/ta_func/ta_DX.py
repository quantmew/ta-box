import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
from .ta_utility import TA_GLOBALS_UNSTABLE_PERIOD, TA_FuncUnstId
from ..settings import TA_FUNC_NO_RANGE_CHECK

if not cython.compiled:
    from .ta_utility import TA_INTEGER_DEFAULT

if not cython.compiled:
    from math import fabs

if not cython.compiled:
    from .ta_utility import TA_IS_ZERO

# Define TRUE_RANGE macro
def TRUE_RANGE(
    th: cython.double, tl: cython.double, yc: cython.double
) -> cython.double:
    tr: cython.double = th - tl
    tempReal2: cython.double = fabs(th - yc)
    if tempReal2 > tr:
        tr = tempReal2
    tempReal2 = fabs(tl - yc)
    if tempReal2 > tr:
        tr = tempReal2
    return tr

def round_pos(x: cython.double) -> cython.double:
    return x


def TA_DX_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """
    TA_DX_Lookback - Directional Movement Index Lookback

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
    if optInTimePeriod > 1:
        return optInTimePeriod + TA_GLOBALS_UNSTABLE_PERIOD(
            TA_FuncUnstId.TA_FUNC_UNST_DX
        )
    else:
        return 2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def TA_INT_DX(
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
    """Internal DX implementation without parameter checks"""
    # Insert local variable here
    today: cython.Py_ssize_t
    lookbackTotal: cython.Py_ssize_t
    outIdx: cython.Py_ssize_t
    prevHigh: cython.double
    prevLow: cython.double
    prevClose: cython.double
    prevMinusDM: cython.double
    prevPlusDM: cython.double
    prevTR: cython.double
    tempReal: cython.double
    tempReal2: cython.double
    diffP: cython.double
    diffM: cython.double
    minusDI: cython.double
    plusDI: cython.double
    i: cython.Py_ssize_t

    """
    The DM1 (one period) is base on the largest part of
    today's range that is outside of yesterdays range.
    
    The following 7 cases explain how the +DM and -DM are
    calculated on one period:
    
    [Cases explanation from the original C code]
    
    When calculating the DM over a period > 1, the one-period DM
    for the desired period are initialy sum. In other word, 
    for a -DM14, sum the -DM1 for the first 14 days (that's 
    13 values because there is no DM for the first day!)
    Subsequent DM are calculated using the Wilder's
    smoothing approach:
    
                                    Previous -DM14
    Today's -DM14 = Previous -DM14 -  -------------- + Today's -DM1
                                         14
    
    Calculation of a -DI14 is as follow:
    
                -DM14
        -DI14 =  --------
                   TR14
    
    Calculation of the TR14 is:
    
                                   Previous TR14
    Today's TR14 = Previous TR14 - -------------- + Today's TR1
                                         14
    
        The first TR14 is the summation of the first 14 TR1. See the
        TA_TRANGE function on how to calculate the true range.
    
    Calculation of the DX14 is:
    
        diffDI = ABS( (-DI14) - (+DI14) )
        sumDI  = (-DI14) + (+DI14)
    
        DX14 = 100 * (diffDI / sumDI)
    
    Reference:
        New Concepts In Technical Trading Systems, J. Welles Wilder Jr
    """

    # Original implementation from Wilder's book was doing some integer
    # rounding in its calculations.
    #
    # This was understandable in the context that at the time the book
    # was written, most user were doing the calculation by hand.
    #
    # For a computer, rounding is unnecessary (and even problematic when inputs
    # are close to 1).
    #
    # TA-Lib does not do the rounding. Still, if you want to reproduce Wilder's examples,
    # you can comment out the following #undef/#define and rebuild the library.

    if optInTimePeriod > 1:
        lookbackTotal = optInTimePeriod + TA_GLOBALS_UNSTABLE_PERIOD(
            TA_FuncUnstId.TA_FUNC_UNST_DX
        )
    else:
        lookbackTotal = 2

    # Adjust startIdx to account for the lookback period.
    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    # Make sure there is still something to evaluate.
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    # Indicate where the next output should be put in the outReal.
    outIdx = 0

    # Process the initial DM and TR
    outBegIdx[0] = today = startIdx

    prevMinusDM = 0.0
    prevPlusDM = 0.0
    prevTR = 0.0
    today = startIdx - lookbackTotal
    prevHigh = inHigh[today]
    prevLow = inLow[today]
    prevClose = inClose[today]
    i = optInTimePeriod - 1
    while i > 0:  # 修复: 原C代码是i-- > 0，Python中应为i > 0
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
        elif (diffP > 0) and (diffP > diffM):
            # Case 1 and 3: +DM=diffP,-DM=0
            prevPlusDM += diffP

        tr = TRUE_RANGE(prevHigh, prevLow, prevClose)
        prevTR += tr
        prevClose = inClose[today]
        i -= 1

    # Skip the unstable period. Note that this loop must be executed
    # at least ONCE to calculate the first DI.
    i = TA_GLOBALS_UNSTABLE_PERIOD(TA_FuncUnstId.TA_FUNC_UNST_DX) + 1
    while i > 0:  # 修复: 原C代码是i-- != 0，Python中应为i > 0
        # Calculate the prevMinusDM and prevPlusDM
        today += 1
        tempReal = inHigh[today]
        diffP = tempReal - prevHigh  # Plus Delta
        prevHigh = tempReal

        tempReal = inLow[today]
        diffM = prevLow - tempReal  # Minus Delta
        prevLow = tempReal

        prevMinusDM -= prevMinusDM / optInTimePeriod
        prevPlusDM -= prevPlusDM / optInTimePeriod

        if (diffM > 0) and (diffP < diffM):
            # Case 2 and 4: +DM=0,-DM=diffM
            prevMinusDM += diffM
        elif (diffP > 0) and (diffP > diffM):
            # Case 1 and 3: +DM=diffP,-DM=0
            prevPlusDM += diffP

        # Calculate the prevTR
        tr = TRUE_RANGE(prevHigh, prevLow, prevClose)
        prevTR = prevTR - (prevTR / optInTimePeriod) + tr
        prevClose = inClose[today]
        i -= 1

    # Write the first DX output
    if not TA_IS_ZERO(prevTR):
        minusDI = round_pos(100.0 * (prevMinusDM / prevTR))
        plusDI = round_pos(100.0 * (prevPlusDM / prevTR))
        tempReal = minusDI + plusDI
        if not TA_IS_ZERO(tempReal):
            outReal[0] = round_pos(100.0 * (fabs(minusDI - plusDI) / tempReal))
        else:
            outReal[0] = 0.0
    else:
        outReal[0] = 0.0
    outIdx = 1

    while today < endIdx:
        # Calculate the prevMinusDM and prevPlusDM
        today += 1
        tempReal = inHigh[today]
        diffP = tempReal - prevHigh  # Plus Delta
        prevHigh = tempReal

        tempReal = inLow[today]
        diffM = prevLow - tempReal  # Minus Delta
        prevLow = tempReal

        prevMinusDM -= prevMinusDM / optInTimePeriod
        prevPlusDM -= prevPlusDM / optInTimePeriod

        if (diffM > 0) and (diffP < diffM):
            # Case 2 and 4: +DM=0,-DM=diffM
            prevMinusDM += diffM
        elif (diffP > 0) and (diffP > diffM):
            # Case 1 and 3: +DM=diffP,-DM=0
            prevPlusDM += diffP

        # Calculate the prevTR
        tr = TRUE_RANGE(prevHigh, prevLow, prevClose)
        prevTR = prevTR - (prevTR / optInTimePeriod) + tr
        prevClose = inClose[today]

        # Calculate the DX. The value is rounded (see Wilder book).
        if not TA_IS_ZERO(prevTR):
            minusDI = round_pos(100.0 * (prevMinusDM / prevTR))
            plusDI = round_pos(100.0 * (prevPlusDM / prevTR))
            tempReal = minusDI + plusDI
            if not TA_IS_ZERO(tempReal):
                outReal[outIdx] = round_pos(100.0 * (fabs(minusDI - plusDI) / tempReal))
            else:
                outReal[outIdx] = outReal[outIdx - 1]
        else:
            outReal[outIdx] = outReal[outIdx - 1]
        outIdx += 1

    outNBElement[0] = outIdx
    return TA_RetCode.TA_SUCCESS


def TA_DX(
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
    """TA_DX - Directional Movement Index

    Input  = High, Low, Close
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
        if inHigh is None or inLow is None or inClose is None:
            return TA_RetCode.TA_BAD_PARAM
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 14
        elif optInTimePeriod < 2 or optInTimePeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM
        if outReal is None:
            return TA_RetCode.TA_BAD_PARAM
    return TA_INT_DX(
        startIdx,
        endIdx,
        inHigh,
        inLow,
        inClose,
        optInTimePeriod,
        outBegIdx,
        outNBElement,
        outReal,
    )


def DX(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int = 14
) -> np.ndarray:
    """DX(high, low, close[, timeperiod=14])

    Directional Movement Index (Overlap Studies)

    The DX is a technical indicator used to determine whether a market is trending or not.
    It is calculated from the Directional Indicator (DI) values.

    Inputs:
        high: (any ndarray) High prices
        low: (any ndarray) Low prices
        close: (any ndarray) Close prices
    Parameters:
        timeperiod: 14 Number of periods
    Outputs:
        real
    """
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)

    if high.shape[0] != low.shape[0] or high.shape[0] != close.shape[0]:
        raise ValueError("Input arrays must have the same length")

    check_timeperiod(timeperiod)

    length: cython.Py_ssize_t = high.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(high)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_DX_Lookback(timeperiod)

    outReal = np.full_like(high, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_DX(
        0,
        endIdx,
        high[startIdx:],
        low[startIdx:],
        close[startIdx:],
        timeperiod,
        outBegIdx,
        outNBElement,
        outReal[lookback:],
    )

    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal
    return outReal
