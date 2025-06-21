import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
from .ta_utility import TA_GLOBALS_UNSTABLE_PERIOD, TA_FuncUnstId, TA_IS_ZERO, TA_INTEGER_DEFAULT
from ..settings import TA_FUNC_NO_RANGE_CHECK
from .ta_TRANGE import TA_TRANGE, TRANGE
from .ta_SMA import TA_SMA, SMA

def TA_NATR_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """
    TA_NATR_Lookback - Normalized Average True Range Lookback

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
    return optInTimePeriod + TA_GLOBALS_UNSTABLE_PERIOD(TA_FuncUnstId.TA_FUNC_UNST_NATR)

@cython.boundscheck(False)
@cython.wraparound(False)
def TA_NATR(
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
    """TA_NATR - Normalized Average True Range

    Input  = High, Low, Close
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
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        if inHigh is None or inLow is None or inClose is None:
            return TA_RetCode.TA_BAD_PARAM
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 14
        elif optInTimePeriod < 1 or optInTimePeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM
        if outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    outBegIdx[0] = 0
    outNBElement[0] = 0

    # Adjust startIdx to account for the lookback period
    lookbackTotal = TA_NATR_Lookback(optInTimePeriod)
    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    if startIdx > endIdx:
        return TA_RetCode.TA_SUCCESS

    # Trap the case where no smoothing is needed
    if optInTimePeriod <= 1:
        return TA_TRANGE(startIdx, endIdx, inHigh, inLow, inClose, outBegIdx, outNBElement, outReal)

    # Allocate an intermediate buffer for TRANGE
    length = lookbackTotal + (endIdx - startIdx) + 1
    tempBuffer = np.full(length, np.nan, dtype=np.double)

    # Do TRANGE in the intermediate buffer
    outBegIdx1 = np.zeros(1, dtype=np.intp)
    outNbElement1 = np.zeros(1, dtype=np.intp)
    retCode = TA_TRANGE(
        startIdx - lookbackTotal + 1, endIdx, inHigh, inLow, inClose,
        outBegIdx1, outNbElement1, tempBuffer
    )

    if retCode != TA_RetCode.TA_SUCCESS:
        return retCode

    # First value of the ATR is a simple Average of the TRANGE output
    prevATRTemp = np.zeros(1, dtype=np.double)
    retCode = TA_SMA(
        optInTimePeriod - 1, optInTimePeriod - 1, tempBuffer, optInTimePeriod,
        outBegIdx1, outNbElement1, prevATRTemp
    )

    if retCode != TA_RetCode.TA_SUCCESS:
        return retCode

    prevATR = prevATRTemp[0]
    today = optInTimePeriod
    outIdx = TA_GLOBALS_UNSTABLE_PERIOD(TA_FuncUnstId.TA_FUNC_UNST_NATR)

    # Skip the unstable period
    while outIdx != 0:
        prevATR *= optInTimePeriod - 1
        prevATR += tempBuffer[today]
        prevATR /= optInTimePeriod
        today += 1
        outIdx -= 1

    # Now start to write the final NATR in the caller provided outReal
    outReal[0] = 0.0
    tempValue = inClose[today]
    if not TA_IS_ZERO(tempValue):
        outReal[0] = (prevATR / tempValue) * 100.0

    outIdx = 1
    nbATR = (endIdx - startIdx) + 1

    # Calculate the remaining range
    while nbATR > 1:
        prevATR *= optInTimePeriod - 1
        prevATR += tempBuffer[today]
        prevATR /= optInTimePeriod
        today += 1
        
        tempValue = inClose[today]
        if not TA_IS_ZERO(tempValue):
            outReal[outIdx] = (prevATR / tempValue) * 100.0
        else:
            outReal[outIdx] = 0.0
            
        outIdx += 1
        nbATR -= 1

    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx

    return TA_RetCode.TA_SUCCESS

def NATR(
    realHigh: np.ndarray,
    realLow: np.ndarray,
    realClose: np.ndarray,
    timeperiod: int = 14
) -> np.ndarray:
    """NATR(realHigh, realLow, realClose[, timeperiod=14])

    Normalized Average True Range (Volatility Indicators)

    The NATR is calculated as (ATR / Close) * 100, where ATR is the Average True Range.
    Normalization makes the ATR more relevant for long-term analysis and cross-security comparison.

    Inputs:
        realHigh: (any ndarray) High price series
        realLow: (any ndarray) Low price series
        realClose: (any ndarray) Close price series
    Parameters:
        timeperiod: 14 Number of periods for the ATR calculation
    Outputs:
        real
    """
    realHigh = check_array(realHigh)
    realLow = check_array(realLow)
    realClose = check_array(realClose)
    check_timeperiod(timeperiod)

    length: cython.Py_ssize_t = realHigh.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(realHigh)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_NATR_Lookback(timeperiod)

    outReal = np.full_like(realHigh, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_NATR(
        0, endIdx, realHigh[startIdx:], realLow[startIdx:], realClose[startIdx:],
        timeperiod, outBegIdx, outNBElement, outReal[lookback:]
    )

    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal
    return outReal