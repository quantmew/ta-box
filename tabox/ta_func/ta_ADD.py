import cython
import numpy as np
from .ta_utils import check_array, check_begidx2, check_length2
from ..retcode import TA_RetCode

def TA_ADD_Lookback() -> cython.Py_ssize_t:
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_ADD(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal0: cython.double[::1],
    inReal1: cython.double[::1],
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.Py_ssize_t:
    # Parameters check
    if startIdx < 0:
        return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
    if endIdx < 0 or endIdx < startIdx:
        return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
    
    outIdx: cython.Py_ssize_t = 0
    for i in range(startIdx, endIdx + 1):
        outReal[outIdx] = inReal0[i] + inReal1[i]
        outIdx += 1
    
    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx
    
    return TA_RetCode.TA_SUCCESS


def ADD(real0: np.ndarray, real1: np.ndarray) -> np.ndarray:
    """ADD(real0, real1)

    Vector Arithmetic Add (Math Operators)

    Inputs:
        real0: (any ndarray)
        real1: (any ndarray)
    Outputs:
        real
    """
    real0 = check_array(real0)
    real1 = check_array(real1)
    length = check_length2(real0, real1)
    startIdx = check_begidx2(real0, real1)

    endIdx = length - startIdx - 1
    lookback = startIdx + TA_ADD_Lookback()
    outReal = np.full_like(real0, np.nan)
    outBegIdx: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    outNBElement: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    retCode = TA_ADD(0, endIdx, real0[startIdx:], real1[startIdx:], outBegIdx, outNBElement, outReal[lookback:])
    return outReal
