import cython
import numpy as np
from .ta_utils import check_array, check_begidx1
from ..retcode import TA_RetCode

if not cython.compiled:
    from math import asin


def TA_ASIN_Lookback() -> cython.Py_ssize_t:
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_ASIN(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    # Parameters check
    if startIdx < 0:
        return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
    if endIdx < 0 or endIdx < startIdx:
        return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
    
    outIdx: cython.Py_ssize_t = 0
    for i in range(startIdx, endIdx + 1):
        outReal[outIdx] = asin(inReal[i])
        outIdx += 1
    
    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx
    
    return TA_RetCode.TA_SUCCESS


def ASIN(real: np.ndarray):
    """ASIN(real)

    Vector Trigonometric ASIN (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    real = check_array(real)

    outReal = np.full_like(real, np.nan)
    length: cython.int = real.shape[0]

    startIdx: cython.int = check_begidx1(real)
    endIdx: cython.int = length - startIdx - 1
    lookback = startIdx + TA_ASIN_Lookback()

    outBegIdx: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.int64)
    outNBElement: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.int64)

    TA_ASIN(0, endIdx, real[startIdx:], outBegIdx, outNBElement, outReal[lookback:])
    return outReal
