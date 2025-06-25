import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
from .ta_utility import TA_INTEGER_DEFAULT
from ..settings import TA_FUNC_NO_RANGE_CHECK
from .ta_MA import TA_MA, TA_MA_Lookback


def TA_STOCHF_Lookback(
    optInFastK_Period: cython.int,
    optInFastD_Period: cython.int,
    optInFastD_MAType: cython.int,
) -> cython.Py_ssize_t:
    """
    TA_STOCHF_Lookback - Stochastic Fast Lookback

    Input:
        optInFastK_Period: (int) Time period for building the Fast-K line (From 1 to 100000)
        optInFastD_Period: (int) Smoothing period for Fast-D (From 1 to 100000)
        optInFastD_MAType: (int) Type of Moving Average for Fast-D

    Output:
        (int) Number of lookback periods
    """
    if not TA_FUNC_NO_RANGE_CHECK:
        if optInFastK_Period == TA_INTEGER_DEFAULT:
            optInFastK_Period = 5
        elif optInFastK_Period < 1 or optInFastK_Period > 100000:
            return -1

        if optInFastD_Period == TA_INTEGER_DEFAULT:
            optInFastD_Period = 3
        elif optInFastD_Period < 1 or optInFastD_Period > 100000:
            return -1

        if optInFastD_MAType == TA_INTEGER_DEFAULT:
            optInFastD_MAType = 0
        elif optInFastD_MAType < 0 or optInFastD_MAType > 8:
            return -1

    lookbackK = optInFastK_Period - 1
    lookbackFastD = TA_MA_Lookback(optInFastD_Period, optInFastD_MAType)
    return lookbackK + lookbackFastD


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_STOCHF(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inHigh: cython.double[::1],
    inLow: cython.double[::1],
    inClose: cython.double[::1],
    optInFastK_Period: cython.int,
    optInFastD_Period: cython.int,
    optInFastD_MAType: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outFastK: cython.double[::1],
    outFastD: cython.double[::1],
) -> cython.int:
    """
    TA_STOCHF - Stochastic Fast

    Input  = High, Low, Close
    Output = double, double (FastK, FastD)

    Optional Parameters:
    -------------------
    optInFastK_Period: (From 1 to 100000)
        Time period for building the Fast-K line
    optInFastD_Period: (From 1 to 100000)
        Smoothing period for making the Fast-D line
    optInFastD_MAType:
        Type of Moving Average for Fast-D
    """
    # Parameter checks
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        if inHigh is None or inLow is None or inClose is None:
            return TA_RetCode.TA_BAD_PARAM
        if outFastK is None or outFastD is None:
            return TA_RetCode.TA_BAD_PARAM

        if optInFastK_Period == TA_INTEGER_DEFAULT:
            optInFastK_Period = 5
        elif optInFastK_Period < 1 or optInFastK_Period > 100000:
            return TA_RetCode.TA_BAD_PARAM

        if optInFastD_Period == TA_INTEGER_DEFAULT:
            optInFastD_Period = 3
        elif optInFastD_Period < 1 or optInFastD_Period > 100000:
            return TA_RetCode.TA_BAD_PARAM

        if optInFastD_MAType == TA_INTEGER_DEFAULT:
            optInFastD_MAType = 0
        elif optInFastD_MAType < 0 or optInFastD_MAType > 8:
            return TA_RetCode.TA_BAD_PARAM

    # Local variables
    lowest: cython.double
    highest: cython.double
    tmp: cython.double
    diff: cython.double
    outIdx: cython.Py_ssize_t
    lowestIdx: cython.Py_ssize_t
    highestIdx: cython.Py_ssize_t
    lookbackK: cython.Py_ssize_t
    lookbackFastD: cython.Py_ssize_t
    lookbackTotal: cython.Py_ssize_t
    trailingIdx: cython.Py_ssize_t
    today: cython.Py_ssize_t
    i: cython.Py_ssize_t
    retCode: cython.int

    # Calculate lookback periods
    lookbackK = optInFastK_Period - 1
    lookbackFastD = TA_MA_Lookback(optInFastD_Period, optInFastD_MAType)
    lookbackTotal = lookbackK + lookbackFastD

    # Adjust start index if needed
    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    # Prepare temporary buffer for FastK calculation
    length = endIdx - (startIdx - lookbackTotal) + 1
    tempBuffer = np.full(length, np.nan, dtype=np.double)

    # Calculate FastK values
    outIdx = 0
    trailingIdx = startIdx - lookbackTotal
    today = trailingIdx + lookbackK
    lowestIdx = highestIdx = -1
    diff = highest = lowest = 0.0

    while today <= endIdx:
        # Find lowest low in the period
        tmp = inLow[today]
        if lowestIdx < trailingIdx:
            lowestIdx = trailingIdx
            lowest = inLow[lowestIdx]
            i = lowestIdx
            while i <= today:
                tmp = inLow[i]
                if tmp < lowest:
                    lowestIdx = i
                    lowest = tmp
                i += 1
            diff = (highest - lowest) / 100.0
        elif tmp <= lowest:
            lowestIdx = today
            lowest = tmp
            diff = (highest - lowest) / 100.0

        # Find highest high in the period
        tmp = inHigh[today]
        if highestIdx < trailingIdx:
            highestIdx = trailingIdx
            highest = inHigh[highestIdx]
            i = highestIdx
            while i <= today:
                tmp = inHigh[i]
                if tmp > highest:
                    highestIdx = i
                    highest = tmp
                i += 1
            diff = (highest - lowest) / 100.0
        elif tmp >= highest:
            highestIdx = today
            highest = tmp
            diff = (highest - lowest) / 100.0

        # Calculate FastK
        if diff != 0.0:
            tempBuffer[outIdx] = (inClose[today] - lowest) / diff
        else:
            tempBuffer[outIdx] = 0.0
        outIdx += 1
        trailingIdx += 1
        today += 1

    # Calculate FastD by smoothing FastK with moving average
    outBegIdx1 = np.zeros(1, dtype=np.intp)
    outNBElement1 = np.zeros(1, dtype=np.intp)
    retCode = TA_MA(
        0,
        outIdx - 1,
        tempBuffer,
        optInFastD_Period,
        optInFastD_MAType,
        outBegIdx1,
        outNBElement1,
        outFastD,
    )

    if retCode != TA_RetCode.TA_SUCCESS or outNBElement1[0] == 0:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return retCode

    # Copy FastK values to output
    # np.copyto(outFastK, tempBuffer[lookbackFastD : lookbackFastD + outNBElement1[0]])
    for i in range(outNBElement1[0]):
        outFastK[i] = tempBuffer[lookbackFastD + i]

    # Set output indices
    outBegIdx[0] = startIdx
    outNBElement[0] = outNBElement1[0]

    return TA_RetCode.TA_SUCCESS


def STOCHF(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    fastk_period: int = 5,
    fastd_period: int = 3,
    fastd_matype: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    STOCHF(high, low, close[, fastk_period=5, fastd_period=3, fastd_matype=0])

    Stochastic Fast (Overlap Studies)

    The Fast Stochastic is calculated as:
    FASTK = 100 * (close - lowest low) / (highest high - lowest low)
    FASTD = Moving average of FASTK over fastd_period

    Inputs:
        high: (any ndarray) High prices
        low: (any ndarray) Low prices
        close: (any ndarray) Close prices
    Parameters:
        fastk_period: 5 Time period for Fast-K
        fastd_period: 3 Smoothing period for Fast-D
        fastd_matype: 0 Type of moving average (0=SMA, 1=EMA, etc.)
    Outputs:
        fastk, fastd
    """
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)

    if high.shape != low.shape or high.shape != close.shape:
        raise ValueError("Input arrays must have the same shape")

    check_timeperiod(fastk_period)
    check_timeperiod(fastd_period)

    length: cython.Py_ssize_t = high.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(high)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_STOCHF_Lookback(
        fastk_period, fastd_period, fastd_matype
    )

    outFastK = np.full_like(high, np.nan)
    outFastD = np.full_like(high, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_STOCHF(
        0,
        endIdx,
        high[startIdx:],
        low[startIdx:],
        close[startIdx:],
        fastk_period,
        fastd_period,
        fastd_matype,
        outBegIdx,
        outNBElement,
        outFastK[lookback:],
        outFastD[lookback:],
    )

    if retCode != TA_RetCode.TA_SUCCESS:
        return outFastK, outFastD
    return outFastK, outFastD
