import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
from .ta_utility import (
    TA_GLOBALS_UNSTABLE_PERIOD,
    TA_FuncUnstId,
    TA_IS_ZERO,
    TA_INTEGER_DEFAULT,
)
from ..settings import TA_FUNC_NO_RANGE_CHECK
from .ta_SMA import TA_SMA, TA_SMA_Lookback


def TA_ULTOSC_Lookback(
    optInTimePeriod1: cython.int, optInTimePeriod2: cython.int, optInTimePeriod3: cython.int
) -> cython.Py_ssize_t:
    """
    TA_ULTOSC_Lookback - Ultimate Oscillator Lookback

    Input:
        optInTimePeriod1: (int) Number of bars for 1st period (From 1 to 100000)
        optInTimePeriod2: (int) Number of bars for 2nd period (From 1 to 100000)
        optInTimePeriod3: (int) Number of bars for 3rd period (From 1 to 100000)

    Output:
        (int) Number of lookback periods
    """
    if not TA_FUNC_NO_RANGE_CHECK:
        if optInTimePeriod1 == TA_INTEGER_DEFAULT:
            optInTimePeriod1 = 7
        elif optInTimePeriod1 < 1 or optInTimePeriod1 > 100000:
            return -1

        if optInTimePeriod2 == TA_INTEGER_DEFAULT:
            optInTimePeriod2 = 14
        elif optInTimePeriod2 < 1 or optInTimePeriod2 > 100000:
            return -1

        if optInTimePeriod3 == TA_INTEGER_DEFAULT:
            optInTimePeriod3 = 28
        elif optInTimePeriod3 < 1 or optInTimePeriod3 > 100000:
            return -1

    # Lookback for the Ultimate Oscillator is the lookback of the SMA with the longest
    # time period, plus 1 for the True Range.
    max_period = max(optInTimePeriod1, optInTimePeriod2, optInTimePeriod3)
    return TA_SMA_Lookback(max_period) + 1

def calc_terms(day, inLow, inHigh, inClose):
    tempLT = inLow[day]
    tempHT = inHigh[day]
    tempCY = inClose[day - 1]
    trueLow = min(tempLT, tempCY)
    closeMinusTrueLow = inClose[day] - trueLow
    trueRange = tempHT - tempLT
    tempDouble = abs(tempCY - tempHT)
    if tempDouble > trueRange:
        trueRange = tempDouble
    tempDouble = abs(tempCY - tempLT)
    if tempDouble > trueRange:
        trueRange = tempDouble
    return trueLow, trueRange, closeMinusTrueLow

def prime_totals(period, startIdx, inLow, inHigh, inClose):
    aTotal = 0.0
    bTotal = 0.0
    for i in range(startIdx - period + 1, startIdx):
        trueLow, trueRange, closeMinusTrueLow = calc_terms(i, inLow, inHigh, inClose)
        aTotal += closeMinusTrueLow
        bTotal += trueRange
    return aTotal, bTotal

def TA_ULTOSC(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inHigh: cython.double[::1],
    inLow: cython.double[::1],
    inClose: cython.double[::1],
    optInTimePeriod1: cython.int,
    optInTimePeriod2: cython.int,
    optInTimePeriod3: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """TA_ULTOSC - Ultimate Oscillator

    Input  = High, Low, Close
    Output = double

    Optional Parameters
    -------------------
    optInTimePeriod1:(From 1 to 100000)
       Number of bars for 1st period.

    optInTimePeriod2:(From 1 to 100000)
       Number of bars for 2nd period

    optInTimePeriod3:(From 1 to 100000)
       Number of bars for 3rd period
    """
    # Parameter checks
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX

        if optInTimePeriod1 == TA_INTEGER_DEFAULT:
            optInTimePeriod1 = 7
        elif optInTimePeriod1 < 1 or optInTimePeriod1 > 100000:
            return TA_RetCode.TA_BAD_PARAM

        if optInTimePeriod2 == TA_INTEGER_DEFAULT:
            optInTimePeriod2 = 14
        elif optInTimePeriod2 < 1 or optInTimePeriod2 > 100000:
            return TA_RetCode.TA_BAD_PARAM

        if optInTimePeriod3 == TA_INTEGER_DEFAULT:
            optInTimePeriod3 = 28
        elif optInTimePeriod3 < 1 or optInTimePeriod3 > 100000:
            return TA_RetCode.TA_BAD_PARAM

        if inHigh is None or inLow is None or inClose is None or outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    outBegIdx[0] = 0
    outNBElement[0] = 0

    # Ensure that the time periods are ordered from shortest to longest.
    periods = [optInTimePeriod1, optInTimePeriod2, optInTimePeriod3]
    sorted_periods = sorted(periods, reverse=True)
    optInTimePeriod1 = sorted_periods[2]
    optInTimePeriod2 = sorted_periods[1]
    optInTimePeriod3 = sorted_periods[0]

    # Adjust startIdx for lookback period.
    lookbackTotal = TA_ULTOSC_Lookback(
        optInTimePeriod1, optInTimePeriod2, optInTimePeriod3
    )
    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    # Make sure there is still something to evaluate.
    if startIdx > endIdx:
        return TA_RetCode.TA_SUCCESS

    # Prime running totals used in moving averages
    a1Total, b1Total = prime_totals(optInTimePeriod1, startIdx, inLow, inHigh, inClose)
    a2Total, b2Total = prime_totals(optInTimePeriod2, startIdx, inLow, inHigh, inClose)
    a3Total, b3Total = prime_totals(optInTimePeriod3, startIdx, inLow, inHigh, inClose)

    # Calculate oscillator
    today = startIdx
    outIdx = 0
    trailingIdx1 = today - optInTimePeriod1 + 1
    trailingIdx2 = today - optInTimePeriod2 + 1
    trailingIdx3 = today - optInTimePeriod3 + 1

    while today <= endIdx:
        # Add on today's terms
        trueLow, trueRange, closeMinusTrueLow = calc_terms(today, inLow, inHigh, inClose)
        a1Total += closeMinusTrueLow
        a2Total += closeMinusTrueLow
        a3Total += closeMinusTrueLow
        b1Total += trueRange
        b2Total += trueRange
        b3Total += trueRange

        # Calculate the oscillator value for today
        output = 0.0

        if not TA_IS_ZERO(b1Total):
            output += 4.0 * (a1Total / b1Total)
        if not TA_IS_ZERO(b2Total):
            output += 2.0 * (a2Total / b2Total)
        if not TA_IS_ZERO(b3Total):
            output += a3Total / b3Total

        # Remove the trailing terms to prepare for next day
        trueLow, trueRange, closeMinusTrueLow = calc_terms(trailingIdx1, inLow, inHigh, inClose)
        a1Total -= closeMinusTrueLow
        b1Total -= trueRange

        trueLow, trueRange, closeMinusTrueLow = calc_terms(trailingIdx2, inLow, inHigh, inClose)
        a2Total -= closeMinusTrueLow
        b2Total -= trueRange

        trueLow, trueRange, closeMinusTrueLow = calc_terms(trailingIdx3, inLow, inHigh, inClose)
        a3Total -= closeMinusTrueLow
        b3Total -= trueRange

        # Write the output
        outReal[outIdx] = 100.0 * (output / 7.0)

        # Increment indexes
        outIdx += 1
        today += 1
        trailingIdx1 += 1
        trailingIdx2 += 1
        trailingIdx3 += 1

    # All done. Indicate the output limits and return.
    outNBElement[0] = outIdx
    outBegIdx[0] = startIdx

    return TA_RetCode.TA_SUCCESS


def ULTOSC(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    timeperiod1: int = 7,
    timeperiod2: int = 14,
    timeperiod3: int = 28,
) -> np.ndarray:
    """ULTOSC(high, low, close[, timeperiod1=7, timeperiod2=14, timeperiod3=28])

    Ultimate Oscillator (Momentum Indicators)

    The Ultimate Oscillator is a momentum oscillator designed to capture momentum
    across three different timeframes.

    Inputs:
        high: (any ndarray) Input high series
        low: (any ndarray) Input low series
        close: (any ndarray) Input close series
    Parameters:
        timeperiod1: 7 Number of periods for the first time frame
        timeperiod2: 14 Number of periods for the second time frame
        timeperiod3: 28 Number of periods for the third time frame
    Outputs:
        real
    """
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    check_timeperiod(timeperiod1)
    check_timeperiod(timeperiod2)
    check_timeperiod(timeperiod3)

    length: int = high.shape[0]
    startIdx: int = check_begidx1(high)
    endIdx: int = length - startIdx - 1
    lookback: int = startIdx + TA_ULTOSC_Lookback(timeperiod1, timeperiod2, timeperiod3)

    outReal = np.full_like(high, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_ULTOSC(
        0,
        endIdx,
        high[startIdx:],
        low[startIdx:],
        close[startIdx:],
        timeperiod1,
        timeperiod2,
        timeperiod3,
        outBegIdx,
        outNBElement,
        outReal[lookback:],
    )

    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal
    return outReal
