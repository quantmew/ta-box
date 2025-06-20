import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode    
from .ta_utility import TA_GLOBALS_UNSTABLE_PERIOD, TA_FuncUnstId, TA_IS_ZERO, TA_INTEGER_DEFAULT
from ..settings import TA_FUNC_NO_RANGE_CHECK

if not cython.compiled:
    from math import fabs


def round_pos(x: cython.double) -> cython.double:
    return x

def TA_ADX_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """
    TA_ADX_Lookback - Average Directional Movement Index Lookback

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
    return (
        (2 * optInTimePeriod)
        + TA_GLOBALS_UNSTABLE_PERIOD(TA_FuncUnstId.TA_FUNC_UNST_ADX)
        - 1
    )


def TRUE_RANGE(
    th: cython.double, tl: cython.double, yc: cython.double
) -> cython.double:
    """
    Calculate the True Range

    Input:
        th: Today's high
        tl: Today's low
        yc: Yesterday's close

    Output:
        (float) True Range
    """
    tr = th - tl
    temp_real2 = fabs(th - yc)
    if temp_real2 > tr:
        tr = temp_real2
    temp_real2 = fabs(tl - yc)
    if temp_real2 > tr:
        tr = temp_real2
    return tr


def TA_ADX(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inHigh: cython.double[::1],
    inLow: cython.double[::1],
    inClose: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> int:
    """
    TA_ADX - Average Directional Movement Index

    Input  = High, Low, Close
    Output = double

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
        if optInTimePeriod == 0:  # default value handling
            optInTimePeriod = 14
        elif optInTimePeriod < 2 or optInTimePeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM
        if inHigh is None or inLow is None or inClose is None:
            return TA_RetCode.TA_BAD_PARAM
        if outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    lookbackTotal = (
        (2 * optInTimePeriod)
        + TA_GLOBALS_UNSTABLE_PERIOD(TA_FuncUnstId.TA_FUNC_UNST_ADX)
        - 1
    )

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

    # Process the initial DM and TR
    today = startIdx
    outBegIdx[0] = today

    prevMinusDM = 0.0
    prevPlusDM = 0.0
    prevTR = 0.0
    today = startIdx - lookbackTotal
    prevHigh = inHigh[today]
    prevLow = inLow[today]
    prevClose = inClose[today]
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
        elif (diffP > 0) and (diffP > diffM):
            # Case 1 and 3: +DM=diffP,-DM=0
            prevPlusDM += diffP

        tr = TRUE_RANGE(prevHigh, prevLow, prevClose)
        prevTR += tr
        prevClose = inClose[today]

    # Add up all the initial DX
    sumDX = 0.0
    i = optInTimePeriod
    while i > 0:
        i -= 1
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

        # Calculate the DX. The value is rounded (see Wilder book)
        if not TA_IS_ZERO(prevTR):
            minusDI = round_pos(100.0 * (prevMinusDM / prevTR))
            plusDI = round_pos(100.0 * (prevPlusDM / prevTR))
            # This loop is just to accumulate the initial DX
            tempReal = minusDI + plusDI
            if not TA_IS_ZERO(tempReal):
                sumDX += round_pos(100.0 * (fabs(minusDI - plusDI) / tempReal))

    # Calculate the first ADX
    prevADX = round_pos(sumDX / optInTimePeriod)

    # Skip the unstable period
    i = TA_GLOBALS_UNSTABLE_PERIOD(TA_FuncUnstId.TA_FUNC_UNST_ADX)
    while i > 0:
        i -= 1
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

        if not TA_IS_ZERO(prevTR):
            # Calculate the DX. The value is rounded (see Wilder book)
            minusDI = round_pos(100.0 * (prevMinusDM / prevTR))
            plusDI = round_pos(100.0 * (prevPlusDM / prevTR))
            tempReal = minusDI + plusDI
            if not TA_IS_ZERO(tempReal):
                tempReal = round_pos(100.0 * (fabs(minusDI - plusDI) / tempReal))
                # Calculate the ADX
                prevADX = round_pos(
                    ((prevADX * (optInTimePeriod - 1)) + tempReal) / optInTimePeriod
                )

    # Output the first ADX
    outReal[0] = prevADX
    outIdx = 1

    # Calculate and output subsequent ADX
    while today < endIdx:
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

        if not TA_IS_ZERO(prevTR):
            # Calculate the DX. The value is rounded (see Wilder book)
            minusDI = round_pos(100.0 * (prevMinusDM / prevTR))
            plusDI = round_pos(100.0 * (prevPlusDM / prevTR))
            tempReal = minusDI + plusDI
            if not TA_IS_ZERO(tempReal):
                tempReal = round_pos(100.0 * (fabs(minusDI - plusDI) / tempReal))
                # Calculate the ADX
                prevADX = round_pos(
                    ((prevADX * (optInTimePeriod - 1)) + tempReal) / optInTimePeriod
                )

        # Output the ADX
        outReal[outIdx] = prevADX
        outIdx += 1

    outNBElement[0] = outIdx
    return TA_RetCode.TA_SUCCESS


def ADX(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int = 14
) -> np.ndarray:
    """
    ADX(high, low, close[, timeperiod=14])

    Average Directional Movement Index (Overlap Studies)

    Inputs:
        high: (any ndarray) High prices
        low: (any ndarray) Low prices
        close: (any ndarray) Close prices
    Parameters:
        timeperiod: 14
    Outputs:
        real: ADX values
    """
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    check_timeperiod(timeperiod)

    length = high.shape[0]
    startIdx = check_begidx1(high)
    endIdx = length - startIdx - 1
    lookback = startIdx + TA_ADX_Lookback(timeperiod)

    outReal = np.full_like(high, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_ADX(
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
