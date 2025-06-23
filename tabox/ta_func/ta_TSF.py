import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
from .ta_utility import TA_INTEGER_DEFAULT
from ..settings import TA_FUNC_NO_RANGE_CHECK

@cython.boundscheck(False)
@cython.wraparound(False)
def TA_TSF_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """
    TA_TSF_Lookback - Time Series Forecast Lookback

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
    return optInTimePeriod - 1

@cython.boundscheck(False)
@cython.wraparound(False)
def TA_TSF(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """
    TA_TSF - Time Series Forecast

    Input  = double
    Output = double

    Optional Parameters
    -------------------
    optInTimePeriod:(From 2 to 100000)
       Number of period
    """
    # Parameter checks
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 14
        elif optInTimePeriod < 2 or optInTimePeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM
        if inReal is None or outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    # Insert TA function code here.
    """
    Linear Regression is a concept also known as the
    "least squares method" or "best fit." Linear
    Regression attempts to fit a straight line between
    several data points in such a way that distance
    between each data point and the line is minimized.

    For each point, a straight line over the specified
    previous bar period is determined in terms
    of y = b + m*x:

    TA_LINEARREG          : Returns b+m*(period-1)
    TA_LINEARREG_SLOPE    : Returns 'm'
    TA_LINEARREG_ANGLE    : Returns 'm' in degree.
    TA_LINEARREG_INTERCEPT: Returns 'b'
    TA_TSF                : Returns b+m*(period)
    """

    # Adjust startIdx to account for the lookback period.
    lookbackTotal: cython.Py_ssize_t = TA_TSF_Lookback(optInTimePeriod)

    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    # Make sure there is still something to evaluate.
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    outIdx: cython.Py_ssize_t = 0  # Index into the output.
    today: cython.Py_ssize_t = startIdx

    SumX: cython.double = optInTimePeriod * (optInTimePeriod - 1) * 0.5
    SumXSqr: cython.double = optInTimePeriod * (optInTimePeriod - 1) * (2 * optInTimePeriod - 1) / 6
    Divisor: cython.double = SumX * SumX - optInTimePeriod * SumXSqr

    while today <= endIdx:
        SumXY: cython.double = 0
        SumY: cython.double = 0
        i: cython.Py_ssize_t = optInTimePeriod
        while i > 0:
            i -= 1
            tempValue1: cython.double = inReal[today - i]
            SumY += tempValue1
            SumXY += cython.cast(cython.double, i) * tempValue1
        m: cython.double = (optInTimePeriod * SumXY - SumX * SumY) / Divisor
        b: cython.double = (SumY - m * SumX) / cython.cast(cython.double, optInTimePeriod)
        outReal[outIdx] = b + m * cython.cast(cython.double, optInTimePeriod)
        outIdx += 1
        today += 1

    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx

    return TA_RetCode.TA_SUCCESS

def TSF(real: np.ndarray, timeperiod: int = 14) -> np.ndarray:
    """
    TSF(real[, timeperiod=14])

    Time Series Forecast (Overlap Studies)

    The TSF is calculated using linear regression to forecast the value 
    at the end of the specified time period.

    Inputs:
        real: (any ndarray) Input series
    Parameters:
        timeperiod: 14 Number of periods for the forecast
    Outputs:
        real
    """
    real = check_array(real)
    check_timeperiod(timeperiod)

    length: cython.Py_ssize_t = real.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_TSF_Lookback(timeperiod)

    outReal = np.full_like(real, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_TSF(
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