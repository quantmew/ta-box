import cython
import numpy as np
from .ta_utils import check_array, check_begidx1, check_timeperiod
from ..retcode import TA_RetCode
from .ta_VAR import TA_INT_VAR

if not cython.compiled:
    from math import sqrt


def INT_stddev_using_precalc_ma(
    inReal: cython.double[::1],
    inMovAvg: cython.double[::1],
    inMovAvgBegIdx: cython.int,
    inMovAvgNbElement: cython.int,
    timePeriod: cython.int,
    outReal: cython.double[::1],
) -> cython.double[::1]:
    """
    pre-calculated moving average standard deviation

    Inputs:
        inReal: (any ndarray)
        inMovAvg: (any ndarray)
        inMovAvgBegIdx: (int)
        inMovAvgNbElement: (int)
        timePeriod: (int)

    Outputs:
        (any ndarray)
    """
    # Initialize output array
    output = outReal

    # Calculate start and end indices for the sum of squares
    startSum = 1 + inMovAvgBegIdx - timePeriod
    endSum = inMovAvgBegIdx

    # Initialize the sum of squares for the period
    periodTotal2 = 0.0

    # Calculate the sum of squares for the initial period
    for outIdx in range(startSum, endSum):
        tempReal = inReal[outIdx]
        tempReal_squared = tempReal * tempReal
        periodTotal2 += tempReal_squared

    # Calculate the standard deviation for each moving average point
    for outIdx in range(inMovAvgNbElement):
        # Add the new price squared value
        tempReal = inReal[endSum]
        tempReal_squared = tempReal * tempReal
        periodTotal2 += tempReal_squared

        # Calculate the mean of the squared values
        meanValue2 = periodTotal2 / timePeriod

        # Remove the old price squared value
        tempReal = inReal[startSum]
        tempReal_squared = tempReal * tempReal
        periodTotal2 -= tempReal_squared

        # Subtract the squared moving average
        tempReal = inMovAvg[outIdx]
        tempReal_squared = tempReal * tempReal
        meanValue2 -= tempReal_squared

        # Calculate the standard deviation
        if meanValue2 > 0:  # Avoid taking the square root of a negative number
            output[outIdx] = np.sqrt(meanValue2)
        else:
            output[outIdx] = 0.0

        # Update indices
        startSum += 1
        endSum += 1

    return output


def TA_STDDEV_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """TA_STDDEV_Lookback(optInTimePeriod) -> Py_ssize_t

    Standard Deviation Lookback
    """
    if optInTimePeriod == 0:
        optInTimePeriod = 5
    elif optInTimePeriod < 2 or optInTimePeriod > 100000:
        return -1

    return optInTimePeriod - 1


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_STDDEV(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    optInNbDev: cython.double,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    # Parameters check
    if startIdx < 0:
        return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
    if endIdx < 0 or endIdx < startIdx:
        return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX

    if optInTimePeriod == 0:
        optInTimePeriod = 5
    elif optInTimePeriod < 2 or optInTimePeriod > 100000:
        return TA_RetCode.TA_BAD_PARAM

    if optInNbDev == 0:
        optInNbDev = 1.0
    elif optInNbDev < -3.0e37 or optInNbDev > 3.0e37:
        return TA_RetCode.TA_BAD_PARAM

    # Calculate the variance
    retCode = TA_INT_VAR(
        startIdx, endIdx, inReal, optInTimePeriod, outBegIdx, outNBElement, outReal
    )
    if retCode != TA_RetCode.TA_SUCCESS:
        return retCode

    # Calculate the square root of each variance, this is the standard deviation
    if optInNbDev != 1.0:
        for i in range(outNBElement[0]):
            tempReal = outReal[i]
            if tempReal > 0:
                outReal[i] = sqrt(tempReal) * optInNbDev
            else:
                outReal[i] = 0.0
    else:
        for i in range(outNBElement[0]):
            tempReal = outReal[i]
            if tempReal > 0:
                outReal[i] = sqrt(tempReal)
            else:
                outReal[i] = 0.0

    return TA_RetCode.TA_SUCCESS


def STDDEV(real: np.ndarray, timeperiod: int = 5, nbdev: float = 1.0):
    """STDDEV(real[, timeperiod=5, nbdev=1.0])

    Standard Deviation (Statistic Functions)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 5
        nbdev: 1.0
    Outputs:
        real
    """
    real = check_array(real)
    check_timeperiod(timeperiod)

    outReal = np.full_like(real, np.nan)
    length: cython.Py_ssize_t = real.shape[0]

    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback = startIdx + TA_STDDEV_Lookback(timeperiod)

    outBegIdx: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    outNBElement: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)

    TA_STDDEV(
        0,
        endIdx,
        real[startIdx:],
        timeperiod,
        nbdev,
        outBegIdx,
        outNBElement,
        outReal[lookback:],
    )
    return outReal
