
cdef extern from "math.h":
    cpdef double sqrt(double x)

cpdef Py_ssize_t TA_SQRT_Lookback()
cpdef int TA_SQRT(Py_ssize_t startIdx, Py_ssize_t endIdx, double[::1] inReal, double[::1] outReal)