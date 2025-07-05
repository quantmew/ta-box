import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
from .ta_utility import TA_GLOBALS_UNSTABLE_PERIOD, TA_FuncUnstId
from ..settings import TA_FUNC_NO_RANGE_CHECK
from .ta_SMA import TA_SMA, TA_SMA_Lookback

if not cython.compiled:
    from .ta_utility import TA_INTEGER_DEFAULT
    from math import fabs
    fmin = min

def TA_IS_ZERO(v: cython.double) -> cython.bint:
    return ((-0.00000001) < v) and (v < 0.00000001)


def TA_ULTOSC_Lookback(
    optInTimePeriod1: cython.int,
    optInTimePeriod2: cython.int,
    optInTimePeriod3: cython.int,
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

@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.wraparound(False)
def calc_terms(
    day: cython.Py_ssize_t,
    inLow: cython.double[::1],
    inHigh: cython.double[::1],
    inClose: cython.double[::1],
) -> tuple[cython.double, cython.double, cython.double]:
    tempLT: cython.double = inLow[day]
    tempHT: cython.double = inHigh[day]
    tempCY: cython.double = inClose[day - 1]
    trueLow: cython.double = min(tempLT, tempCY)
    closeMinusTrueLow: cython.double = inClose[day] - trueLow
    trueRange: cython.double = tempHT - tempLT
    tempDouble: cython.double = abs(tempCY - tempHT)
    if tempDouble > trueRange:
        trueRange = tempDouble
    tempDouble = abs(tempCY - tempLT)
    if tempDouble > trueRange:
        trueRange = tempDouble
    return trueLow, trueRange, closeMinusTrueLow

@cython.cfunc
@cython.inline
def prime_totals(
    period: cython.int,
    startIdx: cython.Py_ssize_t,
    inLow: cython.double[::1],
    inHigh: cython.double[::1],
    inClose: cython.double[::1],
) -> tuple[cython.double, cython.double]:
    aTotal: cython.double = 0.0
    bTotal: cython.double = 0.0

    trueLow: cython.double = 0.0
    trueRange: cython.double = 0.0
    closeMinusTrueLow: cython.double = 0.0

    i: cython.Py_ssize_t = 0
    for i in range(startIdx - period + 1, startIdx):
        if cython.compiled:
            CALC_TERMS(i, trueLow, trueRange, closeMinusTrueLow, inLow, inHigh, inClose)
        else:
            trueLow, trueRange, closeMinusTrueLow = calc_terms(i, inLow, inHigh, inClose)
        aTotal += closeMinusTrueLow
        bTotal += trueRange
    return aTotal, bTotal

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
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
    periods: cython.int[3]
    if not cython.compiled:
        periods = np.array([optInTimePeriod1, optInTimePeriod2, optInTimePeriod3], dtype=int)
    periods[0] = optInTimePeriod1
    periods[1] = optInTimePeriod2
    periods[2] = optInTimePeriod3
    i: cython.Py_ssize_t = 0
    j: cython.Py_ssize_t = 0
    for i in range(3):
        for j in range(i + 1, 3):
            if periods[i] < periods[j]:
                periods[i], periods[j] = periods[j], periods[i]

    optInTimePeriod1 = periods[2]
    optInTimePeriod2 = periods[1]
    optInTimePeriod3 = periods[0]

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
    a1Total: cython.double
    a2Total: cython.double
    a3Total: cython.double
    b1Total: cython.double
    b2Total: cython.double
    b3Total: cython.double

    a1Total, b1Total = prime_totals(optInTimePeriod1, startIdx, inLow, inHigh, inClose)
    a2Total, b2Total = prime_totals(optInTimePeriod2, startIdx, inLow, inHigh, inClose)
    a3Total, b3Total = prime_totals(optInTimePeriod3, startIdx, inLow, inHigh, inClose)

    # Calculate oscillator
    today: cython.Py_ssize_t = startIdx
    outIdx: cython.Py_ssize_t = 0
    trailingIdx1: cython.Py_ssize_t = today - optInTimePeriod1 + 1
    trailingIdx2: cython.Py_ssize_t = today - optInTimePeriod2 + 1
    trailingIdx3: cython.Py_ssize_t = today - optInTimePeriod3 + 1

    output: cython.double = 0.0

    trueLow: cython.double = 0.0
    trueRange: cython.double = 0.0
    closeMinusTrueLow: cython.double = 0.0

    while today <= endIdx:
        # Add on today's terms
        # trueLow, trueRange, closeMinusTrueLow = calc_terms(
        #     today, inLow, inHigh, inClose
        # )
        if cython.compiled:
            CALC_TERMS(today, trueLow, trueRange, closeMinusTrueLow, inLow, inHigh, inClose)
        else:
            trueLow, trueRange, closeMinusTrueLow = calc_terms(
                today, inLow, inHigh, inClose
            )
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
        if cython.compiled:
            CALC_TERMS(trailingIdx1, trueLow, trueRange, closeMinusTrueLow, inLow, inHigh, inClose)
        else:
            trueLow, trueRange, closeMinusTrueLow = calc_terms(
                trailingIdx1, inLow, inHigh, inClose
            )
        a1Total -= closeMinusTrueLow
        b1Total -= trueRange

        if cython.compiled:
            CALC_TERMS(trailingIdx2, trueLow, trueRange, closeMinusTrueLow, inLow, inHigh, inClose)
        else:
            trueLow, trueRange, closeMinusTrueLow = calc_terms(
                trailingIdx2, inLow, inHigh, inClose
            )
        a2Total -= closeMinusTrueLow
        b2Total -= trueRange

        if cython.compiled:
            CALC_TERMS(trailingIdx3, trueLow, trueRange, closeMinusTrueLow, inLow, inHigh, inClose)
        else:
            trueLow, trueRange, closeMinusTrueLow = calc_terms(
                trailingIdx3, inLow, inHigh, inClose
            )
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

    length: cython.int = high.shape[0]
    startIdx: cython.int = check_begidx1(high)
    endIdx: cython.int = length - startIdx - 1
    lookback: cython.int = startIdx + TA_ULTOSC_Lookback(timeperiod1, timeperiod2, timeperiod3)

    outReal: cython.double[::1] = np.full_like(high, np.nan)
    outBegIdx: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    outNBElement: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)

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
