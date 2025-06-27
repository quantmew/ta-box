import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
if not cython.compiled:
    from .ta_utility import TA_INTEGER_DEFAULT
from .ta_EMA import TA_EMA, TA_EMA_Lookback, TA_INT_EMA
from .ta_ROC import TA_ROC, TA_ROC_Lookback
from ..settings import TA_FUNC_NO_RANGE_CHECK


def TA_TRIX_Lookback(optInTimePeriod: cython.Py_ssize_t) -> cython.Py_ssize_t:
    """
    TA_TRIX_Lookback - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA Lookback

    Input:
        optInTimePeriod: (int) Number of period (From 1 to 100000)

    Output:
        (int) Number of lookback periods
    """
    if not TA_FUNC_NO_RANGE_CHECK:
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 30
        elif optInTimePeriod < 1 or optInTimePeriod > 100000:
            return -1

    emaLookback = TA_EMA_Lookback(optInTimePeriod)
    return (emaLookback * 3) + TA_ROC_Lookback(1)


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_INT_TRIX(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """Internal TRIX implementation without parameter checks"""
    k: cython.double
    tempBuffer = np.full(endIdx - startIdx + 1, np.nan, dtype=np.double)
    nbElement: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    begIdx: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    totalLookback: cython.Py_ssize_t
    emaLookback: cython.Py_ssize_t
    rocLookback: cython.Py_ssize_t
    retCode: cython.int
    nbElementToOutput: cython.Py_ssize_t

    emaLookback = TA_EMA_Lookback(optInTimePeriod)
    rocLookback = TA_ROC_Lookback(1)
    totalLookback = (emaLookback * 3) + rocLookback

    if startIdx < totalLookback:
        startIdx = totalLookback

    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    outBegIdx[0] = startIdx
    nbElementToOutput = (endIdx - startIdx) + 1 + totalLookback

    # Calculate the first EMA
    k = 2.0 / (optInTimePeriod + 1)
    retCode = TA_INT_EMA(
        startIdx - totalLookback,
        endIdx,
        inReal,
        optInTimePeriod,
        k,
        begIdx,
        nbElement,
        tempBuffer,
    )

    if retCode != TA_RetCode.TA_SUCCESS:
        outNBElement[0] = 0
        return retCode

    nbElementToOutput -= 1
    nbElementToOutput -= emaLookback

    # Calculate the second EMA
    retCode = TA_INT_EMA(
        0,
        nbElementToOutput,
        tempBuffer,
        optInTimePeriod,
        k,
        begIdx,
        nbElement,
        tempBuffer,
    )

    if retCode != TA_RetCode.TA_SUCCESS:
        outNBElement[0] = 0
        return retCode

    nbElementToOutput -= emaLookback

    # Calculate the third EMA
    retCode = TA_INT_EMA(
        0,
        nbElementToOutput,
        tempBuffer,
        optInTimePeriod,
        k,
        begIdx,
        nbElement,
        tempBuffer,
    )

    if retCode != TA_RetCode.TA_SUCCESS:
        outNBElement[0] = 0
        return retCode

    # Calculate the 1-day Rate-Of-Change
    nbElementToOutput -= emaLookback
    retCode = TA_ROC(0, nbElementToOutput, tempBuffer, 1, begIdx, outNBElement, outReal)

    if retCode != TA_RetCode.TA_SUCCESS or outNBElement[0] == 0:
        outNBElement[0] = 0
        return retCode

    return TA_RetCode.TA_SUCCESS


def TA_TRIX(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """TA_TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA

    Input  = double
    Output = double

    Optional Parameters
    -------------------
    optInTimePeriod:(From 1 to 100000)
       Number of period
    """
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 30
        elif optInTimePeriod < 1 or optInTimePeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM
        if inReal is None or outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    return TA_INT_TRIX(
        startIdx, endIdx, inReal, optInTimePeriod, outBegIdx, outNBElement, outReal
    )


def TRIX(real: np.ndarray, timeperiod: int = 30) -> np.ndarray:
    """TRIX(real[, timeperiod=30])

    1-day Rate-Of-Change (ROC) of a Triple Smooth EMA (Overlap Studies)

    Inputs:
        real: (any ndarray) Input series
    Parameters:
        timeperiod: 30 Number of periods
    Outputs:
        real
    """
    real = check_array(real)
    check_timeperiod(timeperiod)

    length: cython.Py_ssize_t = real.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_TRIX_Lookback(timeperiod)

    outReal = np.full_like(real, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_TRIX(
        0,
        endIdx,
        real[startIdx:],
        timeperiod,
        outBegIdx,
        outNBElement,
        outReal[lookback:],
    )
    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal
    return outReal
