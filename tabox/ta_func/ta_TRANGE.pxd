
cdef extern from "math.h":
    cpdef double fabs(double x)

cpdef Py_ssize_t TA_TRANGE_Lookback()
cpdef int TA_TRANGE(Py_ssize_t startIdx, Py_ssize_t endIdx, double[::1] inHigh, double[::1] inLow, double[::1] inClose, double[::1] outReal)