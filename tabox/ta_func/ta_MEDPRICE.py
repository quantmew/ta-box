import cython
import numpy as np
from .ta_utils import check_array, check_begidx1
from ..retcode import TA_RetCode
from ..settings import TA_FUNC_NO_RANGE_CHECK


def TA_MEDPRICE_Lookback() -> cython.Py_ssize_t:
    """
    TA_MEDPRICE_Lookback - Median Price Lookback

    Output:
        (int) Number of lookback periods (always 0 for MEDPRICE)
    """
    # This function does not require lookback
    return 0


def TA_MEDPRICE(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inHigh: cython.double[::1],
    inLow: cython.double[::1],
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """TA_MEDPRICE - Median Price

    Input  = High, Low
    Output = double

    Calculate median price, formula is (High + Low) / 2
    """
    # Parameter check
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        if inHigh is None or inLow is None:
            return TA_RetCode.TA_BAD_PARAM
        if outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    outIdx: cython.Py_ssize_t = 0
    i: cython.Py_ssize_t

    # MEDPRICE = (High + Low) / 2
    # This is the highest price and lowest price of the same price bar
    # For multiple price bars, see MIDPRICE

    for i in range(startIdx, endIdx + 1):
        outReal[outIdx] = (inHigh[i] + inLow[i]) / 2.0
        outIdx += 1

    outNBElement[0] = outIdx
    outBegIdx[0] = startIdx

    return TA_RetCode.TA_SUCCESS


def MEDPRICE(inHigh: np.ndarray, inLow: np.ndarray, timeperiod: int = 30) -> np.ndarray:
    """MEDPRICE(inHigh, inLow[, timeperiod=30])

    Median Price (Overlap Studies)

    Calculate median price, formula is (High + Low) / 2

    Inputs:
        inHigh: (any ndarray) High price array
        inLow: (any ndarray) Low price array
    Parameters:
        timeperiod: Not used, only for compatibility (default: 30)
    Outputs:
        real: Median price array
    """
    # Check input array
    inHigh = check_array(inHigh)
    inLow = check_array(inLow)

    if inHigh.shape[0] != inLow.shape[0]:
        raise ValueError("inHigh and inLow must have the same length")

    length: cython.Py_ssize_t = inHigh.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(inHigh)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_MEDPRICE_Lookback()

    outReal = np.full_like(inHigh, np.nan)
    outBegIdx = np.zeros(1, dtype=np.int64)
    outNBElement = np.zeros(1, dtype=np.int64)

    retCode = TA_MEDPRICE(
        0,
        endIdx,
        inHigh[startIdx:],
        inLow[startIdx:],
        outBegIdx,
        outNBElement,
        outReal[lookback:],
    )
    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal
    return outReal
