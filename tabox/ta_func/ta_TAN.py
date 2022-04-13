import cython
import numpy as np
from .ta_utils import check_array, check_begidx1

def TA_TAN_Lookback() -> cython.int:
    return 0

def TA_TAN(startIdx: cython.int, endIdx: cython.int, inReal: cython.double[::1], outReal: cython.double[::1]) -> None:
    outReal = np.sqrt(inReal[startIdx:endIdx])

def TAN(real: np.ndarray) -> np.ndarray:
    """ TAN(real)

    Vector Trigonometric Tan (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    real = check_array(real)

    outReal = np.full_like(real, np.nan)
    length: cython.int = real.shape[0]

    startIdx: cython.int = check_begidx1(real)
    endIdx: cython.int = length - 1
    lookback = startIdx + TA_TAN_Lookback()
    return TA_TAN(startIdx, endIdx, real[startIdx:], outReal[lookback:])