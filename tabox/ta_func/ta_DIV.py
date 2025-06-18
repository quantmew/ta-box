
import cython
import numpy as np
from .ta_utils import check_array, check_begidx2, check_length2
from ..retcode import TA_RetCode

def TA_DIV_Lookback() -> cython.int:
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
def TA_DIV(startIdx: cython.int, endIdx: cython.int, inReal0: cython.double[::1], inReal1: cython.double[::1], outReal: cython.double[::1]) -> cython.int:
    outIdx: cython.int = 0
    for i in range(startIdx, endIdx+1):
        outReal[outIdx] = inReal0[i] / inReal1[i]
        outIdx += 1
    
    return TA_RetCode.TA_SUCCESS

def DIV(real0: np.ndarray, real1: np.ndarray) -> np.ndarray:
    """ DIV(real0, real1)

    Vector Arithmetic Div (Math Operators)

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
    lookback = startIdx + TA_DIV_Lookback()
    outreal = np.full_like(real0, np.nan)
    retCode = TA_DIV(0, endIdx, real0[startIdx:], real1[startIdx:], outreal[lookback:])
    return outreal 