
import cython
import numpy as np
from .ta_utils import check_array, check_begidx1

def TA_SMA_Lookback(optInTimePeriod: cython.int):
    return optInTimePeriod - 1

def TA_SMA(startIdx: cython.int, endIdx: cython.int, inReal: np.array, optInTimePeriod: cython.int, outReal: np.array) -> None:
    
    lookbackTotal: cython.int = optInTimePeriod - 1

    if startIdx < lookbackTotal:
        startIdx = lookbackTotal
    
    if startIdx > endIdx:
        return
    
    periodTotal: cython.int = 0
    trailingIdx: cython.int = startIdx - lookbackTotal
   
    i: cython.int = trailingIdx
    if optInTimePeriod > 1:
        while i < startIdx:
            periodTotal += inReal[i]
            i += 1

    outIdx: cython.int = 0
    while True:
        periodTotal += inReal[i]
        i += 1
        tempReal = periodTotal
        periodTotal -= inReal[trailingIdx]
        trailingIdx += 1
        outReal[outIdx] = tempReal / optInTimePeriod
        outIdx += 1

        if not (i <= endIdx):
            break


def SMA(real: np.ndarray, timeperiod: int = 30) -> np.ndarray:
    """ SMA(real[, timeperiod=?])

    Simple Moving Average (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    real = check_array(real)

    outReal = np.full_like(real, np.nan)
    length = real.shape[0]

    startIdx: cython.int = check_begidx1(real)
    endIdx: cython.int = length - 1
    lookback: cython.int = startIdx + TA_SMA_Lookback(timeperiod)

    TA_SMA(startIdx, endIdx, real, timeperiod, outReal[lookback:])

    return outReal
