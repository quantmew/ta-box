import cython
import numpy as np
from .ta_utils import check_array, check_begidx1
from ..retcode import TA_RetCode
from ..settings import TA_FUNC_NO_RANGE_CHECK

def TA_AVGPRICE_Lookback() -> cython.Py_ssize_t:
    """
    TA_AVGPRICE_Lookback - Average Price Lookback

    Output:
        (int) Number of lookback periods (0, as no lookback is needed)
    """
    return 0

def TA_AVGPRICE(
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
    """
    TA_AVGPRICE - Average Price

    Input  = Open, High, Low, Close
    Output = double

    Parameters:
    -----------
    startIdx: Starting index for calculation
    endIdx: Ending index for calculation
    inOpen: Input array of open prices
    inHigh: Input array of high prices
    inLow: Input array of low prices
    inClose: Input array of close prices
    outBegIdx: Output begin index
    outNBElement: Number of output elements
    outReal: Output array of average prices

    Returns:
    --------
    TA_RetCode: Return code indicating success or failure
    """
    # 参数检查
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        if inOpen is None or inHigh is None or inLow is None or inClose is None:
            return TA_RetCode.TA_BAD_PARAM
        if outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    outIdx: cython.Py_ssize_t
    i: cython.Py_ssize_t

    outIdx = 0
    for i in range(startIdx, endIdx + 1):
        outReal[outIdx] = (inHigh[i] + inLow[i] + inClose[i] + inOpen[i]) / 4
        outIdx += 1

    outNBElement[0] = outIdx
    outBegIdx[0] = startIdx

    return TA_RetCode.TA_SUCCESS

def AVGPRICE(
    inOpen: np.ndarray,
    inHigh: np.ndarray,
    inLow: np.ndarray,
    inClose: np.ndarray,
) -> np.ndarray:
    """
    AVGPRICE(inOpen, inHigh, inLow, inClose)

    Average Price

    Inputs:
        inOpen: (any ndarray) Input array of open prices
        inHigh: (any ndarray) Input array of high prices
        inLow: (any ndarray) Input array of low prices
        inClose: (any ndarray) Input array of close prices
    Outputs:
        real: Array of average prices
    """
    # 检查输入数组
    inOpen = check_array(inOpen)
    inHigh = check_array(inHigh)
    inLow = check_array(inLow)
    inClose = check_array(inClose)
    
    # 确保所有输入数组长度一致
    if (inOpen.shape[0] != inHigh.shape[0] or 
        inOpen.shape[0] != inLow.shape[0] or 
        inOpen.shape[0] != inClose.shape[0]):
        raise ValueError("Input arrays must have the same length")

    length: cython.Py_ssize_t = inOpen.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(inOpen)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_AVGPRICE_Lookback()

    outReal = np.full_like(inOpen, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_AVGPRICE(
        0,
        endIdx,
        inOpen[startIdx:],
        inHigh[startIdx:],
        inLow[startIdx:],
        inClose[startIdx:],
        outBegIdx,
        outNBElement,
        outReal[lookback:],
    )
    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal
    return outReal