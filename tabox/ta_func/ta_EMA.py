import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1

from .ta_SMA import TA_SMA_Lookback

def TA_EMA_Lookback(optInTimePeriod: cython.int, optInMAType: cython.int) -> cython.int:
    """ TA_EMA_Lookback(optInTimePeriod) -> int

    EMA Lookback
    """
    return optInTimePeriod - 1

def TA_EMA(startIdx: cython.int, endIdx: cython.int, inReal: cython.double[::1], optInTimePeriod: cython.int, outReal: cython.double[::1]) -> None:
    pass

def EMA(real: np.ndarray, timeperiod: int):
    """ MA(real[, timeperiod=?, matype=?])

    Exponential Moving average (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    real = check_array(real)
    check_timeperiod(timeperiod)

    length: cython.int = real.shape[0]

    startIdx: cython.int = check_begidx1(real)
    endIdx: cython.int = length - startIdx - 1
    lookback: cython.int = startIdx + TA_EMA_Lookback(timeperiod)

    outReal = np.full_like(real, np.nan)

    retCode = TA_EMA(0, endIdx, real[startIdx:], timeperiod, outReal[lookback:])
    return outReal