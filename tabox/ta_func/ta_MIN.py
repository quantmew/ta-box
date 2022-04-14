from sys import prefix
import cython
import numpy as np
from .ta_utils import check_array, check_begidx1, check_timeperiod

def min_double(left: cython.double, right: cython.double) -> cython.double:
    """ min_double(left, right) -> double

    Return the minimum of two doubles
    """
    if left < right:
        return left
    else:
        return right

def TA_MIN_Lookback(optInTimePeriod: cython.int) -> cython.int:
    """ TA_MIN_Lookback(optInTimePeriod) -> int

    MIN Lookback

    Inputs:
        optInTimePeriod: (int)
    Outputs:
        int
    """
    return optInTimePeriod - 1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def TA_MIN_LARGE_TIMEPERIOD(startIdx: cython.int, endIdx: cython.int, inReal: cython.double[::1], optInTimePeriod: cython.int, outReal: cython.double[::1]) -> None:
    n: cython.int = endIdx - startIdx + 1
    prefixMin: cython.double[::1] = np.empty(shape=(n,), dtype=np.float64) # [0] * n
    suffixMin: cython.double[::1] = np.empty(shape=(n,), dtype=np.float64)

    i: cython.int = 0
    inIdx: cython.int = startIdx
    while i < n:
        if i % optInTimePeriod == 0:
            prefixMin[i] = inReal[inIdx]
        else:
            prefixMin[i] = min_double(prefixMin[i-1], inReal[inIdx])
        i += 1
        inIdx += 1

    i: cython.int = n-1
    inIdx: cython.int = startIdx + i
    while i >= 0:
        if i == n - 1 or (i + 1) % optInTimePeriod == 0:
            suffixMin[i] = inReal[inIdx]
        else:
            suffixMin[i] = min_double(suffixMin[i+1], inReal[inIdx])
        i -= 1
        inIdx -= 1

    suffix_i: cython.int = 0
    prefix_i: cython.int = optInTimePeriod - 1
    while suffix_i < n - optInTimePeriod + 1:
        outReal[suffix_i] = min_double(suffixMin[suffix_i], prefixMin[prefix_i])
        suffix_i += 1
        prefix_i += 1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def TA_MIN_SMALL_TIMEPERIOD(startIdx: cython.int, endIdx: cython.int, inReal: cython.double[::1], optInTimePeriod: cython.int, outReal: cython.double[::1]) -> None:
    nbInitialElementNeeded: cython.int = optInTimePeriod-1

    if startIdx < nbInitialElementNeeded:
        startIdx = nbInitialElementNeeded

    if startIdx > endIdx:
        return

    outIdx: cython.int = 0
    today: cython.int = startIdx
    trailingIdx: cython.int = startIdx - nbInitialElementNeeded
    lowestIdx: cython.int = -1
    lowest: cython.double = 0.0

    while today <= endIdx:
        tmp: cython.double = inReal[today]

        if lowestIdx < trailingIdx:
            lowestIdx = trailingIdx
            lowest = inReal[lowestIdx]
            i: cython.int = lowestIdx
            while i + 1 <= today:
                i += 1
                tmp = inReal[i]
                if tmp < lowest:
                    lowestIdx = i
                    lowest = tmp
        elif tmp <= lowest:
            lowestIdx = today
            lowest = tmp
        outReal[outIdx] = lowest
        outIdx += 1
        trailingIdx+=1
        today+=1

def TA_MIN(startIdx: cython.int, endIdx: cython.int, inReal: cython.double[::1], optInTimePeriod: cython.int, outReal: cython.double[::1]) -> None:
    if optInTimePeriod < 100:
        TA_MIN_SMALL_TIMEPERIOD(startIdx, endIdx, inReal, optInTimePeriod, outReal)
    else:
        TA_MIN_LARGE_TIMEPERIOD(startIdx, endIdx, inReal, optInTimePeriod, outReal)

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
    length: cython.int = real.shape[0]
    startIdx: cython.int = check_begidx1(real)
    endIdx: cython.int = length - startIdx - 1
    lookback: cython.int = startIdx + TA_MIN_Lookback(timeperiod)
    outReal = np.full_like(real, np.nan)
    TA_MIN(0, endIdx, real[startIdx:], timeperiod, outReal[lookback:])
    return outReal