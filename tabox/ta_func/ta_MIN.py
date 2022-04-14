from sys import prefix
import cython
import numpy as np
from .ta_utils import check_array, check_begidx1, check_timeperiod


def TA_MIN_Lookback(optInTimePeriod: cython.int) -> cython.int:
    """ TA_MIN_Lookback(optInTimePeriod) -> int

    MIN Lookback

    Inputs:
        optInTimePeriod: (int)
    Outputs:
        int
    """
    return optInTimePeriod - 1

def TA_MIN(startIdx: cython.int, endIdx: cython.int, inReal: cython.double[::1], optInTimePeriod: cython.int, outReal: cython.double[::1]) -> None:
    n: cython.int = endIdx - startIdx + 1
    prefixMin = np.zeros((n,))
    suffixMin = np.zeros((n,))

    for i in range(n):
        if i % optInTimePeriod == 0:
            prefixMin[i] = inReal[startIdx + i]
        else:
            prefixMin[i] = min(prefixMin[i-1], inReal[startIdx + i])

    for i in range(n-1, -1, -1):
        if i == n - 1 or (i + 1) % optInTimePeriod == 0:
            suffixMin[i] = inReal[startIdx + i]
        else:
            suffixMin[i] = min(suffixMin[i+1], inReal[startIdx + i])

    for i in range(n - optInTimePeriod + 1):
        outReal[i] = min(suffixMin[i], prefixMin[i + optInTimePeriod - 1])

def MIN(real: np.ndarray, timeperiod: cython.int) -> np.ndarray:
    """ MIN(real[, timeperiod=?])

    Highest value over a specified period (Math Operators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    real = check_array(real)
    length = real.shape[0]
    startIdx = check_begidx1(real)
    endIdx= length - startIdx - 1
    lookback = startIdx + TA_MIN_Lookback(timeperiod)
    outReal = np.full_like(real, np.nan)
    TA_MIN(0, endIdx, real[startIdx:], timeperiod, outReal[lookback:])
    return outReal