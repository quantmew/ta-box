import cython
import numpy as np
from .ta_utils import check_array, check_begidx3, check_length3, make_double_array
from .ta_TRANGE import TA_TRANGE

def TA_ATR_Lookback(optInTimePeriod: int) -> int:
    return optInTimePeriod

@cython.boundscheck(False)
@cython.wraparound(False)
def TA_ATR(startIdx: cython.int, endIdx: cython.int, inHigh: cython.double[::1], inLow: cython.double[::1], inClose: cython.double[::1], optInTimePeriod: cython.int, outReal: cython.double[::1]) -> None:
    lookbackTotal: cython.int = TA_ATR_Lookback(optInTimePeriod)
    if startIdx < lookbackTotal:
      startIdx = lookbackTotal

    if startIdx > endIdx:
        return

    if optInTimePeriod <= 1:
        return TA_TRANGE(startIdx, endIdx, inHigh, inLow, inClose, outReal)

    tempBuffer = np.zeros((lookbackTotal+(endIdx-startIdx)+1,), dtype=float)

    TA_TRANGE(startIdx-lookbackTotal+1, endIdx, inHigh, inLow, inClose, tempBuffer)

    # INT_SMA(optInTimePeriod-1, optInTimePeriod-1, tempBuffer, optInTimePeriod, prevATRTemp)

    # prevATR = prevATRTemp[0]

    # Subsequent value are smoothed using the
    # previous ATR value (Wilder's approach).
    # 1) Multiply the previous ATR by 'period-1'. 
    # 2) Add today TR value. 
    # 3) Divide by 'period'.
    #
    '''
    today = optInTimePeriod
    outIdx = TA_GLOBALS_UNSTABLE_PERIOD(TA_FUNC_UNST_ATR,Atr)
    while outIdx != 0 :
        prevATR *= optInTimePeriod - 1
        prevATR += tempBuffer[today]
        today += 1
        prevATR /= optInTimePeriod
        outIdx -= 1

    outIdx = 1
    outReal[0] = prevATR

    nbATR = (endIdx - startIdx)+1

    while nbATR - 1 != 0:
        nbATR -= 1
        prevATR *= optInTimePeriod - 1
        prevATR += tempBuffer[today]
        today += 1
        prevATR /= optInTimePeriod
        outReal[outIdx] = prevATR
        outIdx += 1
    '''

def ATR(high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int) -> np.ndarray:
    """ ATR(high, low, close[, timeperiod=?])

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
    TA_ATR(0, endIdx, high[startIdx:], low[startIdx:], close[startIdx:], timeperiod, outreal[lookback:])
    return outreal 
