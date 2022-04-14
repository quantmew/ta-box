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

def check_timeperiod(timeperiod: cython.int) -> None:
    if timeperiod <= 1:
        raise Exception('function failed with error code 2: Bad Parameter (TA_BAD_PARAM)')

def check_begidx1(a1: cython.double[::1]) -> cython.int:
    length = a1.shape[0]
    for i in range(length):
        val: cython.double = a1[i]
        if np.isnan(val):
            continue
        return i
    else:
        raise Exception("inputs are all NaN")

def check_begidx2(a1: cython.double[::1], a2: cython.double[::1]) -> cython.int:
    length = a1.shape[0]
    for i in range(length):
        val = a1[i]
        if np.isnan(val):
            continue
        val = a2[i]
        if np.isnan(val):
            continue
        return i
    raise Exception("inputs are all NaN")

def check_length2(a1: cython.double[::1], a2: cython.double[::1]):
    length = a1.shape[0]
    if length != a2.shape[0]:
        raise Exception("input array lengths are different")
    return length