import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
from .ta_utility import TA_IS_ZERO_OR_NEG

from ..settings import TA_FUNC_NO_RANGE_CHECK

if not cython.compiled:
    from .ta_utility import TA_INTEGER_DEFAULT

def TA_CORREL_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """
    TA_CORREL_Lookback - Pearson's Correlation Coefficient Lookback

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
    return optInTimePeriod - 1

@cython.boundscheck(False)
@cython.wraparound(False)
def TA_CORREL(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal0: cython.double[::1],
    inReal1: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """
    TA_CORREL - Pearson's Correlation Coefficient (r)

    Input  = double, double
    Output = double

    Optional Parameters
    -------------------
    optInTimePeriod:(From 1 to 100000)
       Number of period
    """
    # Parameter checks
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        if inReal0 is None or inReal1 is None:
            return TA_RetCode.TA_BAD_PARAM
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 30
        elif optInTimePeriod < 1 or optInTimePeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM
        if outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    # Calculate lookback and adjust start index
    lookbackTotal: cython.Py_ssize_t = optInTimePeriod - 1
    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    # Check if there's data to process
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    outBegIdx[0] = startIdx
    trailingIdx: cython.Py_ssize_t = startIdx - lookbackTotal

    # Initialize sums
    sumXY: cython.double = 0.0
    sumX: cython.double = 0.0
    sumY: cython.double = 0.0
    sumX2: cython.double = 0.0
    sumY2: cython.double = 0.0
    x: cython.double
    y: cython.double
    today: cython.Py_ssize_t

    # Calculate initial sums
    for today in range(trailingIdx, startIdx + 1):
        x = inReal0[today]
        sumX += x
        sumX2 += x * x

        y = inReal1[today]
        sumXY += x * y
        sumY += y
        sumY2 += y * y

    # Calculate first correlation coefficient
    trailingX: cython.double = inReal0[trailingIdx]
    trailingY: cython.double = inReal1[trailingIdx]
    tempReal: cython.double = (sumX2 - (sumX * sumX) / optInTimePeriod) * (sumY2 - (sumY * sumY) / optInTimePeriod)
    if not TA_IS_ZERO_OR_NEG(tempReal):
        outReal[0] = (sumXY - (sumX * sumY) / optInTimePeriod) / np.sqrt(tempReal)
    else:
        outReal[0] = 0.0

    # Process remaining values
    outIdx: cython.Py_ssize_t = 1
    today = startIdx + 1
    trailingIdx += 1

    while today <= endIdx:
        # Remove trailing values
        sumX -= trailingX
        sumX2 -= trailingX * trailingX

        sumXY -= trailingX * trailingY
        sumY -= trailingY
        sumY2 -= trailingY * trailingY

        # Add new values
        x = inReal0[today]
        sumX += x
        sumX2 += x * x

        y = inReal1[today]
        sumXY += x * y
        sumY += y
        sumY2 += y * y

        # Calculate next correlation coefficient
        trailingX = inReal0[trailingIdx]
        trailingY = inReal1[trailingIdx]
        tempReal = (sumX2 - (sumX * sumX) / optInTimePeriod) * (sumY2 - (sumY * sumY) / optInTimePeriod)
        if not TA_IS_ZERO_OR_NEG(tempReal):
            outReal[outIdx] = (sumXY - (sumX * sumY) / optInTimePeriod) / np.sqrt(tempReal)
        else:
            outReal[outIdx] = 0.0

        outIdx += 1
        today += 1
        trailingIdx += 1

    outNBElement[0] = outIdx
    return TA_RetCode.TA_SUCCESS

def CORREL(real0: np.ndarray, real1: np.ndarray, timeperiod: int = 30) -> np.ndarray:
    """
    CORREL(real0, real1[, timeperiod=30])

    Pearson's Correlation Coefficient (r) (Momentum Indicators)

    The Pearson's correlation coefficient measures the linear dependence between two variables X and Y.
    It has a value between +1 and −1, where 1 is total positive linear correlation,
    0 is no linear correlation, and −1 is total negative linear correlation.

    Inputs:
        real0: (any ndarray) First input series
        real1: (any ndarray) Second input series
    Parameters:
        timeperiod: 30 Number of periods
    Outputs:
        real
    """
    real0 = check_array(real0)
    real1 = check_array(real1)
    check_timeperiod(timeperiod)

    if real0.shape[0] != real1.shape[0]:
        raise ValueError("Input arrays must have the same length")

    length: cython.Py_ssize_t = real0.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(real0)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_CORREL_Lookback(timeperiod)

    outReal = np.full_like(real0, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_CORREL(
        0,
        endIdx,
        real0[startIdx:],
        real1[startIdx:],
        timeperiod,
        outBegIdx,
        outNBElement,
        outReal[lookback:],
    )
    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal
    return outReal