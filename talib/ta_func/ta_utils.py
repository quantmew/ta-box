import numpy as np
import cython

def check_array(real: np.ndarray) -> np.ndarray:
    if real.dtype != np.float64:
        raise Exception("input array type is not double")
    if real.ndim != 1:
        raise Exception("input array has wrong dimensions")
    if not real.data.c_contiguous:
        real = np.ascontiguousarray(real)
    return real

def check_begidx1(a1: np.array) -> cython.int:
    length = a1.shape[0]
    for i in range(length):
        val = a1[i]
        if np.isnan(val):
            continue
        return i
    else:
        raise Exception("inputs are all NaN")