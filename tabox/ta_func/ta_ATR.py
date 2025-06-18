import cython
import numpy as np
from .ta_utils import check_array, check_begidx3, check_length3, make_double_array
from .ta_TRANGE import TA_TRANGE
from .ta_SMA import TA_INT_SMA
from ..retcode import *


def TA_ATR_Lookback(optInTimePeriod: cython.Py_ssize_t) -> cython.Py_ssize_t:
    return optInTimePeriod


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_ATR(
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
    lookbackTotal: cython.Py_ssize_t = TA_ATR_Lookback(optInTimePeriod)
    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_SUCCESS

    if optInTimePeriod <= 1:
        return TA_TRANGE(startIdx, endIdx, inHigh, inLow, inClose, outReal)

    tempBuffer = np.zeros((lookbackTotal + (endIdx - startIdx) + 1,), dtype=float)
    prevATRTemp = np.zeros((1,), dtype=float)
    outBegIdx1 = np.zeros(1, dtype=np.int64)
    outNBElement1 = np.zeros(1, dtype=np.int64)

    retCode = TA_TRANGE(startIdx - lookbackTotal + 1, endIdx, inHigh, inLow, inClose, tempBuffer)
    if retCode != TA_SUCCESS:
        return retCode

    # Calculate first ATR value using SMA
    retCode = TA_INT_SMA(
        optInTimePeriod - 1,
        optInTimePeriod - 1,
        tempBuffer,
        optInTimePeriod,
        outBegIdx1,
        outNBElement1,
        prevATRTemp,
    )
    if retCode != TA_SUCCESS:
        return retCode

    prevATR = prevATRTemp[0]

    # Handle unstable period
    today = optInTimePeriod
    outIdx = 0
    unstablePeriod = 1  # TA_GLOBALS_UNSTABLE_PERIOD(TA_FUNC_UNST_ATR,Atr)
    
    # Skip the unstable period
    while unstablePeriod != 0:
        prevATR *= optInTimePeriod - 1
        prevATR += tempBuffer[today]
        today += 1
        prevATR /= optInTimePeriod
        unstablePeriod -= 1

    # Write the first ATR value
    outReal[0] = prevATR
    outIdx = 1

    # Calculate remaining ATR values
    nbATR = (endIdx - startIdx) + 1
    while nbATR - 1 != 0:
        nbATR -= 1
        prevATR *= optInTimePeriod - 1
        prevATR += tempBuffer[today]
        today += 1
        prevATR /= optInTimePeriod
        outReal[outIdx] = prevATR
        outIdx += 1

    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx
    return TA_SUCCESS


def ATR(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int = 14
) -> np.ndarray:
    """ATR(high, low, close[, timeperiod=?])

    Average True Range (Volatility Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """

    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    startIdx = check_begidx3(high, low, close)
    endIdx = length - startIdx - 1
    lookback = startIdx + TA_ATR_Lookback(timeperiod)
    outreal = make_double_array(length, lookback)
    
    outBegIdx = np.zeros(1, dtype=np.int64)
    outNBElement = np.zeros(1, dtype=np.int64)
    
    TA_ATR(
        0,
        endIdx,
        high[startIdx:],
        low[startIdx:],
        close[startIdx:],
        timeperiod,
        outBegIdx,
        outNBElement,
        outreal[lookback:],
    )
    return outreal
