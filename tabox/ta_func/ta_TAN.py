import cython
import numpy as np
from .ta_utils import check_array, check_begidx1
from ..retcode import TA_RetCode

if not cython.compiled:
    from math import tan


def TA_TAN_Lookback() -> cython.Py_ssize_t:
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_TAN(
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
        outReal[outIdx] = tan(inReal[i])
        outIdx += 1
    
    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx
    return TA_RetCode.TA_SUCCESS


def TAN(real: np.ndarray) -> np.ndarray:
    """TAN(real)

    Vector Trigonometric Tan (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    real = check_array(real)

    outReal = np.full_like(real, np.nan)
    length: cython.Py_ssize_t = real.shape[0]

    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback = startIdx + TA_TAN_Lookback()
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)
    retCode = TA_TAN(0, endIdx, real[startIdx:], outBegIdx, outNBElement, outReal[lookback:])
    if retCode != TA_RetCode.TA_SUCCESS:
        raise RuntimeError(f"TA_TAN failed with error code {retCode}")
    return outReal
