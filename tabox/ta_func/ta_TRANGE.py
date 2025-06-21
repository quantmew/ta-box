import numpy as np
import cython
from .ta_utils import check_array, check_begidx1
from ..retcode import TA_RetCode
from ..settings import TA_FUNC_NO_RANGE_CHECK


def TA_TRANGE_Lookback() -> cython.Py_ssize_t:
    """
    TA_TRANGE_Lookback - True Range Lookback

    Output:
        (int) Number of lookback periods
    """
    # No parameter validation needed
    return 1


def TA_TRANGE(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inHigh: cython.double[::1],
    inLow: cython.double[::1],
    inClose: cython.double[::1],
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> int:
    """
    TA_TRANGE - True Range

    Input  = High, Low, Close
    Output = double

    True Range is the maximum of the following three values:
    - The difference between today's high and low (val1)
    - The absolute value of the difference between yesterday's close and today's high (val2)
    - The absolute value of the difference between yesterday's close and today's low (val3)

    Note: To avoid inconsistencies, this function ignores the first price bar and only outputs valid values from the second price bar onwards.
    """
    # Parameter check
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if (endIdx < 0) or (endIdx < startIdx):
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        if inHigh is None or inLow is None or inClose is None:
            return TA_RetCode.TA_BAD_PARAM
        if outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    # If the start index is less than 1, adjust it to 1 (at least two price bars are required)
    if startIdx < 1:
        startIdx = 1

    # Ensure there is data to calculate
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    outIdx = 0
    today = startIdx
    while today <= endIdx:
        # Calculate three possible volatility values
        tempLT = inLow[today]
        tempHT = inHigh[today]
        tempCY = inClose[today - 1]
        greatest = tempHT - tempLT  # val1: Today's high-low difference

        val2 = abs(
            tempCY - tempHT
        )  # val2: Absolute value of yesterday's close to today's high difference
        if val2 > greatest:
            greatest = val2

        val3 = abs(
            tempCY - tempLT
        )  # val3: Absolute value of yesterday's close to today's low difference
        if val3 > greatest:
            greatest = val3

        outReal[outIdx] = greatest
        outIdx += 1
        today += 1

    outNBElement[0] = outIdx
    outBegIdx[0] = startIdx
    return TA_RetCode.TA_SUCCESS


def TRANGE(
    inHigh: np.ndarray,
    inLow: np.ndarray,
    inClose: np.ndarray,
    fastperiod: int = 14,  # This parameter is only for compatibility, TRANGE does not require a period parameter
) -> np.ndarray:
    """
    TRANGE(inHigh, inLow, inClose[, fastperiod=14])

    True Range (Overlap Studies)

    Calculate the true range, which is the maximum of the following three values:
    - The difference between today's high and low
    - The absolute value of the difference between yesterday's close and today's high
    - The absolute value of the difference between yesterday's close and today's low

    Inputs:
        inHigh: (ndarray) High price sequence
        inLow: (ndarray) Low price sequence
        inClose: (ndarray) Close price sequence
    Parameters:
        fastperiod: 14 (This parameter is not used in TRANGE, it is only for compatibility)
    Outputs:
        real: True range sequence
    """
    # Check input arrays
    inHigh = check_array(inHigh)
    inLow = check_array(inLow)
    inClose = check_array(inClose)

    length = inHigh.shape[0]
    if length != inLow.shape[0] or length != inClose.shape[0]:
        raise ValueError("Input array lengths must be consistent")

    startIdx = check_begidx1(inHigh)
    endIdx = length - startIdx - 1
    lookback = startIdx + TA_TRANGE_Lookback()

    outReal = np.full_like(inHigh, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_TRANGE(
        0,
        endIdx,
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
