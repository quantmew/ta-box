import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
if not cython.compiled:
    from .ta_utility import TA_GLOBALS_UNSTABLE_PERIOD, TA_FuncUnstId, TA_INTEGER_DEFAULT
from ..settings import TA_FUNC_NO_RANGE_CHECK
from .ta_ADX import TA_ADX, TA_ADX_Lookback

def TA_ADXR_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """
    TA_ADXR_Lookback - Average Directional Movement Index Rating Lookback

    Input:
        optInTimePeriod: (int) Number of period (From 2 to 100000)

    Output:
        (int) Number of lookback periods
    """
    if not TA_FUNC_NO_RANGE_CHECK:
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 14
        elif optInTimePeriod < 2 or optInTimePeriod > 100000:
            return -1
    
    # Calculate lookback based on ADX lookback and time period
    return optInTimePeriod + TA_ADX_Lookback(optInTimePeriod) - 1

def TA_ADXR(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inHigh: cython.double[::1],
    inLow: cython.double[::1],
    inClose: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1]
) -> int:
    """TA_ADXR - Average Directional Movement Index Rating

    Input  = High, Low, Close
    Output = double

    Optional Parameters
    -------------------
    optInTimePeriod:(From 2 to 100000)
       Number of period
    """
    # Parameter validation
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        if optInTimePeriod == 0:
            optInTimePeriod = 14
        elif optInTimePeriod < 2 or optInTimePeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM
        if inHigh is None or inLow is None or inClose is None or outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    # Adjust startIdx if there is not enough initial data
    lookback = TA_ADXR_Lookback(optInTimePeriod)
    if startIdx < lookback:
        startIdx = lookback

    # Check if there is still data to process
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    # Calculate ADX values first
    adx_length: cython.Py_ssize_t = endIdx - startIdx + optInTimePeriod
    adx: cython.double[::1] = np.zeros(adx_length)
    adx_begIdx: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    adx_nbElement: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)

    retCode = TA_ADX(
        startIdx - (optInTimePeriod - 1),
        endIdx,
        inHigh,
        inLow,
        inClose,
        optInTimePeriod,
        adx_begIdx,
        adx_nbElement,
        adx
    )

    if retCode != TA_RetCode.TA_SUCCESS:
        return retCode

    # Calculate ADXR values from ADX values
    i: cython.Py_ssize_t = optInTimePeriod - 1
    j: cython.Py_ssize_t = 0
    outIdx: cython.Py_ssize_t = 0
    nbElement: cython.Py_ssize_t = endIdx - startIdx + 1  # Fix: should be +1 instead of +2 as in C code
    
    while nbElement > 0:
        outReal[outIdx] = (adx[i] + adx[j]) / 2.0
        i += 1
        j += 1
        outIdx += 1
        nbElement -= 1

    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx

    return TA_RetCode.TA_SUCCESS

def ADXR(high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int = 14) -> np.ndarray:
    """ADXR(high, low, close[, timeperiod=14])

    Average Directional Movement Index Rating (Overlap Studies)

    Inputs:
        high: (np.ndarray) High prices
        low: (np.ndarray) Low prices
        close: (np.ndarray) Close prices
    Parameters:
        timeperiod: 14
    Outputs:
        real: (np.ndarray) ADXR values
    """
    # Validate input arrays
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    
    # Validate timeperiod
    check_timeperiod(timeperiod)

    length: cython.Py_ssize_t = high.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(high)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_ADXR_Lookback(timeperiod)

    # Initialize output array with NaN values
    outReal: cython.double[::1] = np.full_like(high, np.nan)
    outBegIdx: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    outNBElement: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)

    # Calculate ADXR
    retCode = TA_ADXR(
        0,
        endIdx,
        high[startIdx:],
        low[startIdx:],
        close[startIdx:],
        timeperiod,
        outBegIdx,
        outNBElement,
        outReal[lookback:]
    )

    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal

    return outReal