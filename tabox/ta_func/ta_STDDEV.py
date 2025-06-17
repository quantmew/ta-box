import cython
import numpy as np
from .ta_utils import check_array, check_begidx1, check_timeperiod
from ..retcode import *
from .ta_VAR import TA_INT_VAR

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
        return TA_OUT_OF_RANGE_START_INDEX
    if endIdx < 0 or endIdx < startIdx:
        return TA_OUT_OF_RANGE_END_INDEX

    if optInTimePeriod == 0:
        optInTimePeriod = 5
    elif optInTimePeriod < 2 or optInTimePeriod > 100000:
        return TA_BAD_PARAM

    if optInNbDev == 0:
        optInNbDev = 1.0
    elif optInNbDev < -3.0e37 or optInNbDev > 3.0e37:
        return TA_BAD_PARAM

    # Calculate the variance
    retCode = TA_INT_VAR(startIdx, endIdx, inReal, optInTimePeriod, outBegIdx, outNBElement, outReal)
    if retCode != TA_SUCCESS:
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

    return TA_SUCCESS

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

    outBegIdx: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.int64)
    outNBElement: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.int64)

    TA_STDDEV(0, endIdx, real[startIdx:], timeperiod, nbdev,
              outBegIdx, outNBElement, outReal[lookback:])
    return outReal 