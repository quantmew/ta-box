import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
if not cython.compiled:
    from .ta_utility import TA_INTEGER_DEFAULT
from ..settings import TA_FUNC_NO_RANGE_CHECK


def TA_CCI_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """
    TA_CCI_Lookback - Commodity Channel Index Lookback

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
@cython.cdivision(True)
def TA_CCI(
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
    """TA_CCI - Commodity Channel Index

    Input  = High, Low, Close (double)
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
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 14
        elif optInTimePeriod < 2 or optInTimePeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM
        if inHigh is None or inLow is None or inClose is None or outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    tempReal: cython.double
    tempReal2: cython.double
    theAverage: cython.double
    lastValue: cython.double
    i: cython.Py_ssize_t = 0
    j: cython.Py_ssize_t = 0
    lookbackTotal: cython.Py_ssize_t

    # Identify the minimum number of price bar needed to calculate at least one output.
    lookbackTotal = optInTimePeriod - 1

    # Move up the start index if there is not enough initial data.
    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    # Make sure there is still something to evaluate.
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    # Allocate a circular buffer equal to the requested period.
    circBuffer: cython.double[::1] = np.full(optInTimePeriod, 0.0, dtype=np.double)
    circBuffer_Idx: cython.Py_ssize_t = 0

    # Add-up the initial period, except for the last value. Fill up the circular buffer at the same time.
    i = startIdx - lookbackTotal
    if optInTimePeriod > 1:
        while i < startIdx:
            circBuffer[circBuffer_Idx] = (inHigh[i] + inLow[i] + inClose[i]) / 3
            i += 1
            circBuffer_Idx = (circBuffer_Idx + 1) % optInTimePeriod

    # Proceed with the calculation for the requested range.
    outIdx: cython.Py_ssize_t = 0
    i = startIdx
    while i <= endIdx:
        # Calculate the typical price
        lastValue = (inHigh[i] + inLow[i] + inClose[i]) / 3
        circBuffer[circBuffer_Idx] = lastValue

        # Calculate the average for the whole period.
        theAverage = 0.0
        for j in range(optInTimePeriod):
            theAverage += circBuffer[j]
        theAverage /= optInTimePeriod

        # Do the summation of the ABS(TypePrice-average) for the whole period.
        tempReal2 = 0.0
        for j in range(optInTimePeriod):
            tempReal2 += abs(circBuffer[j] - theAverage)

        # Calculate the CCI
        tempReal = lastValue - theAverage

        if tempReal != 0.0 and tempReal2 != 0.0:
            outReal[outIdx] = tempReal / (0.015 * (tempReal2 / optInTimePeriod))
        else:
            outReal[outIdx] = 0.0

        outIdx += 1

        # Move forward the circular buffer index.
        circBuffer_Idx = (circBuffer_Idx + 1) % optInTimePeriod
        i += 1

    # All done. Indicate the output limits and return.
    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx

    return TA_RetCode.TA_SUCCESS


def CCI(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int = 14
) -> np.ndarray:
    """CCI(high, low, close[, timeperiod=14])

    Commodity Channel Index (Momentum Indicators)

    The Commodity Channel Index (CCI) is a technical analysis indicator used to
    identify overbought or oversold conditions in a security.

    Inputs:
        high: (any ndarray) High prices
        low: (any ndarray) Low prices
        close: (any ndarray) Close prices
    Parameters:
        timeperiod: 14 Number of periods to use for the CCI calculation
    Outputs:
        real
    """
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)

    if high.shape != low.shape or high.shape != close.shape:
        raise ValueError("Input arrays must have the same shape")

    check_timeperiod(timeperiod)

    length: cython.Py_ssize_t = high.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(high)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_CCI_Lookback(timeperiod)

    outReal = np.full_like(high, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_CCI(
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
