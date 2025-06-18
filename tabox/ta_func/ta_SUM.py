
import cython
import numpy as np
from .ta_utils import check_array, check_begidx1, check_timeperiod
from ..retcode import TA_RetCode

def TA_SUM_Lookback(optInTimePeriod: cython.int) -> cython.int:
    return optInTimePeriod - 1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def TA_SUM(startIdx: cython.int, endIdx: cython.int, inReal: cython.double[::1], optInTimePeriod: cython.int, outReal: cython.double[::1]) -> cython.int:
    # Identify the minimum number of price bar needed to calculate at least one output.
    lookbackTotal: cython.int = optInTimePeriod - 1

    # Move up the start index if there is not enough initial data.
    if startIdx < lookbackTotal:
        startIdx = lookbackTotal
    
    # Make sure there is still something to evaluate.
    if startIdx > endIdx:
        return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
    
    # Do the MA calculation using tight loops.
    # Add-up the initial period, except for the last value.
    periodTotal: cython.double = 0.0
    trailingIdx: cython.int = startIdx - lookbackTotal
   
    i: cython.int = trailingIdx
    if optInTimePeriod > 1:
        while i < startIdx:
            periodTotal += inReal[i]
            i += 1

    # Proceed with the calculation for the requested range.
    # Note that this algorithm allows the inReal and outReal to be the same buffer.
    outIdx: cython.int = 0
    while True:
        periodTotal += inReal[i]
        i += 1

        tempReal: cython.double = periodTotal

        periodTotal -= inReal[trailingIdx]
        trailingIdx += 1
        
        outReal[outIdx] = tempReal
        outIdx += 1

        if not (i <= endIdx):
            break
    
    # All done. Indicate the output limits and return.
    return TA_RetCode.TA_SUCCESS

def SUM(real: np.ndarray, timeperiod: int = 30) -> np.ndarray:
    """ SUM(real[, timeperiod=?])

    Summation (Math Operators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    real = check_array(real)
    check_timeperiod(timeperiod)

    outReal = np.full_like(real, np.nan)
    length: cython.int = real.shape[0]

    startIdx: cython.int = check_begidx1(real)
    endIdx: cython.int = length - startIdx - 1
    lookback: cython.int = startIdx + TA_SUM_Lookback(timeperiod)

    TA_SUM(0, endIdx, real[startIdx:], timeperiod, outReal[lookback:])

    return outReal
