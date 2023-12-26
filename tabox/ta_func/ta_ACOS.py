import cython
import numpy as np
from .ta_utils import check_array, check_begidx1
from ..retcode import *

if not cython.compiled:
    from math import acos

def TA_ACOS_Lookback() -> cython.int:
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
def TA_ACOS(startIdx: cython.int, endIdx: cython.int, inReal: cython.double[::1], outReal: cython.double[::1]) -> cython.int:
    outIdx: cython.int = 0
    for i in range(startIdx, endIdx+1):
        outReal[outIdx] = acos(inReal[i])
        outIdx += 1

    return TA_SUCCESS

def ACOS(real: np.ndarray):
    """ ACOS(real)

    Vector Trigonometric ACos (Math Transform)

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
    lookback = startIdx + TA_ACOS_Lookback()

    TA_ACOS(0, endIdx, real[startIdx:], outReal[lookback:])
    return outReal