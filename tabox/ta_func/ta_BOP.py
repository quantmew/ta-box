import cython
import numpy as np
from .ta_utils import check_array, check_begidx1
from ..retcode import TA_RetCode
from ..settings import TA_FUNC_NO_RANGE_CHECK
from .ta_utility import TA_IS_ZERO

def TA_BOP_Lookback() -> cython.Py_ssize_t:
    """
    TA_BOP_Lookback - Balance Of Power Lookback

    Input:
        None

    Output:
        (int) Number of lookback periods
    """
    # No parameters to validate.
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
def TA_BOP(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inOpen: cython.double[::1],
    inHigh: cython.double[::1],
    inLow: cython.double[::1],
    inClose: cython.double[::1],
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """TA_BOP - Balance Of Power

    Input  = Open, High, Low, Close
    Output = double

    Optional Parameters
    -------------------
    None
    """
    # Parameter checks
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        if inOpen is None or inHigh is None or inLow is None or inClose is None or outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    # BOP = (Close - Open)/(High - Low)
    outIdx: cython.Py_ssize_t = 0
    tempReal: cython.double

    i: cython.Py_ssize_t
    for i in range(startIdx, endIdx + 1):
        tempReal = inHigh[i] - inLow[i]
        if TA_IS_ZERO(tempReal) or tempReal < 0:  # Fix: Check for zero or negative denominator
            outReal[outIdx] = 0.0
        else:
            outReal[outIdx] = (inClose[i] - inOpen[i]) / tempReal
        outIdx += 1

    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx
    return TA_RetCode.TA_SUCCESS

def BOP(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """BOP(open, high, low, close)

    Balance Of Power (Momentum Indicators)

    The Balance Of Power is calculated using the formula:
    BOP = (Close - Open) / (High - Low)

    Inputs:
        open: (any ndarray) Input open series
        high: (any ndarray) Input high series
        low: (any ndarray) Input low series
        close: (any ndarray) Input close series
    Outputs:
        real
    """
    open_ = check_array(open_)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)

    length: cython.Py_ssize_t = open_.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(open_)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_BOP_Lookback()

    outReal = np.full_like(open_, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_BOP(
        0, endIdx, open_[startIdx:], high[startIdx:], low[startIdx:], close[startIdx:],
        outBegIdx, outNBElement, outReal[lookback:],
    )
    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal
    return outReal