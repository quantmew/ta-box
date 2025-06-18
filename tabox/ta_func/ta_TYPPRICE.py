
import cython
import numpy as np
from .ta_utils import check_array, check_begidx1
from ..retcode import TA_RetCode
from ..settings import TA_FUNC_NO_RANGE_CHECK


def TA_TYPPRICE_Lookback() -> cython.Py_ssize_t:
    """
    TA_TYPPRICE_Lookback - Typical Price Lookback

    Output:
        (int) Number of lookback periods
    """
    # This function have no lookback needed.
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_INT_TYPPRICE(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inHigh: cython.double[::1],
    inLow: cython.double[::1],
    inClose: cython.double[::1],
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """Internal TYPPRICE implementation without parameter checks"""
    outIdx: cython.Py_ssize_t = 0
    i: cython.Py_ssize_t

    # Typical price = (High + Low + Close ) / 3
    for i in range(startIdx, endIdx + 1):
        outReal[outIdx] = (inHigh[i] + inLow[i] + inClose[i]) / 3.0
        outIdx += 1

    outNBElement[0] = outIdx
    outBegIdx[0] = startIdx
    return TA_RetCode.TA_SUCCESS


def TA_TYPPRICE(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inHigh: cython.double[::1],
    inLow: cython.double[::1],
    inClose: cython.double[::1],
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """TA_TYPPRICE - Typical Price

    Input  = High, Low, Close
    Output = double
    """
    # parameters check
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        if inHigh is None or inLow is None or inClose is None or outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    # Call internal implementation
    return TA_INT_TYPPRICE(
        startIdx, endIdx, inHigh, inLow, inClose, outBegIdx, outNBElement, outReal
    )


def TYPPRICE(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """TYPPRICE(high, low, close)

    Typical Price (Overlap Studies)

    Inputs:
        high: (any ndarray) High price
        low: (any ndarray) Low price
        close: (any ndarray) Close price
    Outputs:
        real
    """
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)

    length: cython.Py_ssize_t = high.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(high)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_TYPPRICE_Lookback()

    outReal = np.full_like(high, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_TYPPRICE(
        0,
        endIdx,
        high[startIdx:],
        low[startIdx:],
        close[startIdx:],
        outBegIdx,
        outNBElement,
        outReal[lookback:],
    )
    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal
    return outReal