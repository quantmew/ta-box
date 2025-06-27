from .ta_utility cimport TA_INTEGER_DEFAULT

cdef extern from "math.h":
    cpdef double fabs(double x)

cpdef Py_ssize_t TA_TRANGE_Lookback()
cpdef int TA_TRANGE(
    Py_ssize_t startIdx,
    Py_ssize_t endIdx,
    double[::1] inHigh,
    double[::1] inLow,
    double[::1] inClose,
    Py_ssize_t[::1] outBegIdx,
    Py_ssize_t[::1] outNBElement,
    double[::1] outReal
)