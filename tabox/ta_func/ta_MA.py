import cython
import numpy as np
from .ta_utils import check_array, check_begidx1
from ..retcode import TA_RetCode
from .ta_SMA import TA_SMA, TA_SMA_Lookback
from .ta_EMA import TA_EMA, TA_EMA_Lookback
from .ta_WMA import TA_WMA, TA_WMA_Lookback
from .ta_DEMA import TA_DEMA, TA_DEMA_Lookback
from .ta_TEMA import TA_TEMA, TA_TEMA_Lookback
from .ta_TRIMA import TA_TRIMA, TA_TRIMA_Lookback
from .ta_KAMA import TA_KAMA, TA_KAMA_Lookback
from .ta_MAMA import TA_MAMA, TA_MAMA_Lookback
from .ta_T3 import TA_T3, TA_T3_Lookback

def TA_MA_Lookback(optInTimePeriod: cython.int, optInMAType: cython.int) -> cython.Py_ssize_t:
    """TA_MA_Lookback(optInTimePeriod, optInMAType) -> Py_ssize_t

    Moving Average Lookback
    """
    if optInTimePeriod == 0:
        optInTimePeriod = 30
    if optInTimePeriod < 1 or optInTimePeriod > 100000:
        return -1

    if optInMAType == 0:
        optInMAType = 0
    elif optInMAType < 0 or optInMAType > 8:
        return -1

    if optInTimePeriod <= 1:
        return 0

    if optInMAType == 0:  # SMA
        return TA_SMA_Lookback(optInTimePeriod)
    elif optInMAType == 1:  # EMA
        return TA_EMA_Lookback(optInTimePeriod)
    elif optInMAType == 2:  # WMA
        return TA_WMA_Lookback(optInTimePeriod)
    elif optInMAType == 3:  # DEMA
        return TA_DEMA_Lookback(optInTimePeriod)
    elif optInMAType == 4:  # TEMA
        return TA_TEMA_Lookback(optInTimePeriod)
    elif optInMAType == 5:  # TRIMA
        return TA_TRIMA_Lookback(optInTimePeriod)
    elif optInMAType == 6:  # KAMA
        return TA_KAMA_Lookback(optInTimePeriod)
    elif optInMAType == 7:  # MAMA
        return TA_MAMA_Lookback(0.5, 0.05)
    elif optInMAType == 8:  # T3
        return TA_T3_Lookback(optInTimePeriod, 0.7)
    else:
        return 0

@cython.boundscheck(False)
@cython.wraparound(False)
def TA_MA(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    optInMAType: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    # Parameters check
    if startIdx < 0:
        return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
    if endIdx < 0 or endIdx < startIdx:
        return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX

    if optInTimePeriod == 0:
        optInTimePeriod = 30
    elif optInTimePeriod < 1 or optInTimePeriod > 100000:
        return TA_RetCode.TA_BAD_PARAM

    if optInMAType == 0:
        optInMAType = 0
    elif optInMAType < 0 or optInMAType > 8:
        return TA_RetCode.TA_BAD_PARAM

    if optInTimePeriod == 1:
        nbElement = endIdx - startIdx + 1
        outNBElement[0] = nbElement
        for i in range(nbElement):
            outReal[i] = inReal[startIdx + i]
        outBegIdx[0] = startIdx
        return TA_RetCode.TA_SUCCESS

    if optInMAType == 0:  # SMA
        return TA_SMA(startIdx, endIdx, inReal, optInTimePeriod, outBegIdx, outNBElement, outReal)
    elif optInMAType == 1:  # EMA
        return TA_EMA(startIdx, endIdx, inReal, optInTimePeriod, outBegIdx, outNBElement, outReal)
    elif optInMAType == 2:  # WMA
        return TA_WMA(startIdx, endIdx, inReal, optInTimePeriod, outBegIdx, outNBElement, outReal)
    elif optInMAType == 3:  # DEMA
        return TA_DEMA(startIdx, endIdx, inReal, optInTimePeriod, outBegIdx, outNBElement, outReal)
    elif optInMAType == 4:  # TEMA
        return TA_TEMA(startIdx, endIdx, inReal, optInTimePeriod, outBegIdx, outNBElement, outReal)
    elif optInMAType == 5:  # TRIMA
        return TA_TRIMA(startIdx, endIdx, inReal, optInTimePeriod, outBegIdx, outNBElement, outReal)
    elif optInMAType == 6:  # KAMA
        return TA_KAMA(startIdx, endIdx, inReal, optInTimePeriod, outBegIdx, outNBElement, outReal)
    elif optInMAType == 7:  # MAMA
        dummyBuffer = np.zeros(endIdx - startIdx + 1, dtype=np.float64)
        return TA_MAMA(startIdx, endIdx, inReal, 0.5, 0.05, outBegIdx, outNBElement, outReal, dummyBuffer)
    elif optInMAType == 8:  # T3
        return TA_T3(startIdx, endIdx, inReal, optInTimePeriod, 0.7, outBegIdx, outNBElement, outReal)
    else:
        return TA_RetCode.TA_BAD_PARAM

def MA(real: np.ndarray, timeperiod: int = 30, matype: int = 0):
    """MA(real, timeperiod=30, matype=0)

    Moving Average

    Inputs:
        real: (any ndarray)
        timeperiod: (int) Number of period
        matype: (int) Type of Moving Average
            0 = Simple Moving Average (SMA)
            1 = Exponential Moving Average (EMA)
            2 = Weighted Moving Average (WMA)
            3 = Double Exponential Moving Average (DEMA)
            4 = Triple Exponential Moving Average (TEMA)
            5 = Triangular Moving Average (TRIMA)
            6 = Kaufman Adaptive Moving Average (KAMA)
            7 = MESA Adaptive Moving Average (MAMA)
            8 = Triple Exponential Moving Average (T3)
    Outputs:
        real: (ndarray) Moving Average
    """
    real = check_array(real)

    outReal = np.full_like(real, np.nan)
    length: cython.Py_ssize_t = real.shape[0]

    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback = startIdx + TA_MA_Lookback(timeperiod, matype)

    outBegIdx: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    outNBElement: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)

    TA_MA(0, endIdx, real[startIdx:], timeperiod, matype,
          outBegIdx, outNBElement, outReal[lookback:])
    return outReal