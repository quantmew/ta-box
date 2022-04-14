import cython
import numpy as np
from .ta_utils import check_array, check_begidx1
if not cython.compiled:
    from math import sqrt

def TA_SQRT_Lookback() -> cython.int:
    return 0

def TA_SQRT(startIdx: cython.int, endIdx: cython.int, inReal: cython.double[::1], outReal: cython.double[::1]) -> None:
    outIdx: cython.int = 0
    for i in range(startIdx, endIdx+1):
        outReal[outIdx] = sqrt(inReal[i])
        outIdx += 1

def SQRT(real: np.ndarray) -> np.ndarray:
    """ SQRT(real)

    Vector Square Root (Math Transform)

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
    lookback = startIdx + TA_SQRT_Lookback()
    TA_SQRT(0, endIdx, real[startIdx:], outReal[lookback:])
    return outReal