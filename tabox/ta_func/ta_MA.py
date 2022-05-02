import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1

from .ta_SMA import TA_SMA_Lookback

def TA_MA_Lookback(optInTimePeriod: cython.int, optInMAType: cython.int) -> cython.int:
    """ TA_MA_Lookback(optInTimePeriod) -> int

    MA Lookback
    """
    if optInMAType == 0:
        return TA_SMA_Lookback(optInTimePeriod)
    return optInTimePeriod - 1

def TA_MA(startIdx: cython.int, endIdx: cython.int, inReal: cython.double[::1], optInTimePeriod: cython.int, optInMAType: cython.int, outReal: cython.double[::1]) -> None:
    pass

def MA(real: np.ndarray, timeperiod: int, matype: int=0):
    """ MA(real[, timeperiod=?, matype=?])

    Moving average (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
        matype: 0 (Simple Moving Average)
    Outputs:
        real
    """
    real = check_array(real)
    check_timeperiod(timeperiod)

    length: cython.int = real.shape[0]

    startIdx: cython.int = check_begidx1(real)
    endIdx: cython.int = length - startIdx - 1
    lookback: cython.int = startIdx + TA_MA_Lookback(timeperiod)

    outReal = np.full_like(real, np.nan)

    retCode = TA_MA(0, endIdx, real[startIdx:], timeperiod, matype, outReal[lookback:])
    return outReal