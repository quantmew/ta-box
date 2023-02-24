
import cython
import numpy as np
from .ta_utils import check_array, check_begidx1, check_timeperiod

def TA_WMA_Lookback(optInTimePeriod: cython.int) -> cython.int:
    return optInTimePeriod - 1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def TA_WMA(startIdx: cython.int, endIdx: cython.int, inReal: cython.double[::1], optInTimePeriod: cython.int, outReal: cython.double[::1]) -> None:
    pass

def WMA(real: np.ndarray, timeperiod: int = 30 ) -> np.ndarray:
    """ WMA(real[, timeperiod=?])

    Weighted Moving Average (Overlap Studies)

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
    lookback: cython.int = startIdx + TA_WMA_Lookback(timeperiod)

    TA_WMA(0, endIdx, real[startIdx:], timeperiod, outReal[lookback:])

    return outReal