import cython
import numpy as np
from .ta_utils import check_array, check_begidx1
from ..retcode import TA_RetCode

def TA_T3_Lookback(optInTimePeriod: cython.int, optInVFactor: cython.double) -> cython.Py_ssize_t:
    """TA_T3_Lookback(optInTimePeriod, optInVFactor) -> Py_ssize_t

    T3 Lookback
    """
    if optInTimePeriod == 0:
        optInTimePeriod = 5
    elif optInTimePeriod < 2 or optInTimePeriod > 100000:
        return -1

    if optInVFactor == 0:
        optInVFactor = 0.7
    elif optInVFactor < 0 or optInVFactor > 1:
        return -1

    return 6 * (optInTimePeriod - 1)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def TA_T3(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    optInVFactor: cython.double,
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

    if optInVFactor == 0:
        optInVFactor = 0.7
    elif optInVFactor < 0 or optInVFactor > 1:
        return TA_RetCode.TA_BAD_PARAM

    lookbackTotal: cython.Py_ssize_t = 6 * (optInTimePeriod - 1)
    if startIdx <= lookbackTotal:
        startIdx = lookbackTotal

    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    outBegIdx[0] = startIdx
    today: cython.Py_ssize_t = startIdx - lookbackTotal

    k: cython.double = 2.0 / (optInTimePeriod + 1.0)
    one_minus_k: cython.double = 1.0 - k

    # Initialize e1
    i: cython.Py_ssize_t = 0
    tempReal: cython.double = inReal[today]
    today += 1
    for i in range(optInTimePeriod - 1):
        tempReal += inReal[today]
        today += 1
    e1: cython.double = tempReal / optInTimePeriod

    # Initialize e2
    tempReal = e1
    for i in range(optInTimePeriod - 1):
        e1 = (k * inReal[today]) + (one_minus_k * e1)
        tempReal += e1
        today += 1
    e2: cython.double = tempReal / optInTimePeriod

    # Initialize e3
    tempReal = e2
    for i in range(optInTimePeriod - 1):
        e1 = (k * inReal[today]) + (one_minus_k * e1)
        e2 = (k * e1) + (one_minus_k * e2)
        tempReal += e2
        today += 1
    e3: cython.double = tempReal / optInTimePeriod

    # Initialize e4
    tempReal = e3
    for i in range(optInTimePeriod - 1):
        e1 = (k * inReal[today]) + (one_minus_k * e1)
        e2 = (k * e1) + (one_minus_k * e2)
        e3 = (k * e2) + (one_minus_k * e3)
        tempReal += e3
        today += 1
    e4: cython.double = tempReal / optInTimePeriod

    # Initialize e5
    tempReal = e4
    for i in range(optInTimePeriod - 1):
        e1 = (k * inReal[today]) + (one_minus_k * e1)
        e2 = (k * e1) + (one_minus_k * e2)
        e3 = (k * e2) + (one_minus_k * e3)
        e4 = (k * e3) + (one_minus_k * e4)
        tempReal += e4 # type: ignore
        today += 1
    e5: cython.double = tempReal / optInTimePeriod

    # Initialize e6
    tempReal = e5
    for i in range(optInTimePeriod - 1):
        e1 = (k * inReal[today]) + (one_minus_k * e1)
        e2 = (k * e1) + (one_minus_k * e2)
        e3 = (k * e2) + (one_minus_k * e3)
        e4 = (k * e3) + (one_minus_k * e4)
        e5 = (k * e4) + (one_minus_k * e5)
        tempReal += e5
        today += 1
    e6: cython.double = tempReal / optInTimePeriod

    # Skip the unstable period
    while today <= startIdx:
        e1 = (k * inReal[today]) + (one_minus_k * e1)
        e2 = (k * e1) + (one_minus_k * e2)
        e3 = (k * e2) + (one_minus_k * e3)
        e4 = (k * e3) + (one_minus_k * e4)
        e5 = (k * e4) + (one_minus_k * e5)
        e6 = (k * e5) + (one_minus_k * e6)
        today += 1

    # Calculate the constants
    tempReal = optInVFactor * optInVFactor
    c1 = -(tempReal * optInVFactor)
    c2 = 3.0 * (tempReal - c1)
    c3 = -6.0 * tempReal - 3.0 * (optInVFactor - c1)
    c4 = 1.0 + 3.0 * optInVFactor - c1 + 3.0 * tempReal

    # Write the first output
    outIdx: cython.Py_ssize_t = 0
    outReal[outIdx] = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
    outIdx += 1

    # Calculate and output the remaining of the range
    while today <= endIdx:
        e1 = (k * inReal[today]) + (one_minus_k * e1)
        e2 = (k * e1) + (one_minus_k * e2)
        e3 = (k * e2) + (one_minus_k * e3)
        e4 = (k * e3) + (one_minus_k * e4)
        e5 = (k * e4) + (one_minus_k * e5)
        e6 = (k * e5) + (one_minus_k * e6)
        outReal[outIdx] = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
        outIdx += 1
        today += 1

    outNBElement[0] = outIdx
    return TA_RetCode.TA_SUCCESS

def T3(real: np.ndarray, timeperiod: int = 5, vfactor: float = 0.7):
    """T3(real, timeperiod=5, vfactor=0.7)

    Triple Exponential Moving Average (T3)

    Inputs:
        real: (any ndarray)
        timeperiod: (int) Number of period
        vfactor: (float) Volume Factor
    Outputs:
        real: (ndarray) T3
    """
    real = check_array(real)

    outReal = np.full_like(real, np.nan)
    length: cython.Py_ssize_t = real.shape[0]

    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback = startIdx + TA_T3_Lookback(timeperiod, vfactor)

    outBegIdx: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    outNBElement: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)

    TA_T3(0, endIdx, real[startIdx:], timeperiod, vfactor,
          outBegIdx, outNBElement, outReal[lookback:])
    return outReal 