import cython
import numpy as np
from .ta_utils import check_array, check_length3, check_begidx3
if not cython.compiled:
    from math import fabs

def TA_TRANGE_Lookback() -> cython.int:
    return 1

@cython.boundscheck(False)
@cython.wraparound(False)
def TA_TRANGE(startIdx: cython.int, endIdx: cython.int, inHigh: cython.double[::1], inLow: cython.double[::1], inClose: cython.double[::1], outReal: cython.double[::1]) -> None:
    if startIdx < 1:
        startIdx = 1
    
    if startIdx > endIdx:
        return
    
    outIdx: cython.int = 0
    today: cython.int = startIdx
    while today <= endIdx:
        tempHT: cython.double = inHigh[today]
        tempLT: cython.double = inLow[today]
        tempCY: cython.double = inClose[today-1]
        greatest: cython.double = tempHT - tempLT; 
        val2: cython.double = fabs(tempCY - tempHT)
        if val2 > greatest:
            greatest = val2
        val3: cython.double = fabs( tempCY - tempLT  )
        if val3 > greatest:
            greatest = val3
        outReal[outIdx] = greatest
        outIdx += 1
        today += 1

def TRANGE(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)

    length = check_length3(high, low, close)
    startIdx: cython.int = check_begidx3(high, low, close)
    endIdx: cython.int = length - startIdx - 1
    lookback = startIdx + TA_TRANGE_Lookback()
    outReal = np.full_like(high, np.nan)
    TA_TRANGE(0, endIdx, high[startIdx:], low[startIdx:], close[startIdx:], outReal[lookback:])
    return outReal