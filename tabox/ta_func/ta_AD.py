import cython
from cython.parallel import prange
import numpy as np
from .ta_utils import check_array, check_begidx1
from ..retcode import TA_RetCode


def TA_AD_Lookback() -> cython.Py_ssize_t:
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_AD(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inHigh: cython.double[::1],
    inLow: cython.double[::1],
    inClose: cython.double[::1],
    inVolume: cython.double[::1],
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.Py_ssize_t:
    # Parameters check
    if startIdx < 0:
        return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
    if endIdx < 0 or endIdx < startIdx:
        return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX

    nbBar: cython.Py_ssize_t = endIdx - startIdx + 1
    currentBar: cython.Py_ssize_t = startIdx
    outIdx: cython.Py_ssize_t = 0
    ad: cython.double = 0.0

    high: cython.double
    low: cython.double
    close: cython.double
    tmp: cython.double

    while nbBar != 0:
        high = inHigh[currentBar]
        low = inLow[currentBar]
        tmp = high - low
        close = inClose[currentBar]

        if tmp > 0.0:
            ad += (((close - low) - (high - close)) / tmp) * inVolume[currentBar]

        outReal[outIdx] = ad
        outIdx += 1
        currentBar += 1
        nbBar -= 1

    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx

    return TA_RetCode.TA_SUCCESS


def AD(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray):
    """AD(high, low, close, volume)

    Chaikin A/D Line (Volume Indicators)

    Inputs:
        high: (any ndarray)
        low: (any ndarray)
        close: (any ndarray)
        volume: (any ndarray)
    Outputs:
        real
    """
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    volume = check_array(volume)

    outReal = np.full_like(high, np.nan)
    length: cython.Py_ssize_t = high.shape[0]

    startIdx: cython.Py_ssize_t = check_begidx1(high)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback = startIdx + TA_AD_Lookback()

    outBegIdx: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.int64)
    outNBElement: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.int64)

    TA_AD(0, endIdx, high[startIdx:], low[startIdx:], close[startIdx:], volume[startIdx:],
          outBegIdx, outNBElement, outReal[lookback:])
    return outReal
