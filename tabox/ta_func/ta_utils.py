from typing import Any
import numpy as np
import cython

def check_array(real: Any) -> np.ndarray:
    if not isinstance(real, np.ndarray):
        real = np.array(real, dtype=np.float64)

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

def check_begidx3(a1: cython.double[::1], a2: cython.double[::1], a3: cython.double[::1]) -> cython.int:
    length = a1.shape[0]
    for i in range(length):
        val = a1[i]
        if np.isnan(val):
            continue
        val = a2[i]
        if np.isnan(val):
            continue
        val = a3[i]
        if np.isnan(val):
            continue
        return i
    raise Exception("inputs are all NaN")

def check_length2(a1: cython.double[::1], a2: cython.double[::1]):
    length = a1.shape[0]
    if length != a2.shape[0]:
        raise Exception("input array lengths are different")
    return length

def check_length3(a1: cython.double[::1], a2: cython.double[::1], a3: cython.double[::1]):
    length = a1.shape[0]
    if length != a2.shape[0]:
        raise Exception("input array lengths are different")
    if length != a3.shape[0]:
        raise Exception("input array lengths are different")
    return length

def make_double_array(length: int, lookback: int) -> np.ndarray:
    outreal = np.empty((length,), dtype=np.float64)
    outreal[:lookback] = np.nan
    return outreal