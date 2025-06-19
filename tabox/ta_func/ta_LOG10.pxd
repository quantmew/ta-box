cdef extern from "math.h":
    cpdef double log10(double x)

cpdef Py_ssize_t TA_LOG10_Lookback()
cpdef int TA_LOG10(Py_ssize_t startIdx, Py_ssize_t endIdx, double[::1] inReal, Py_ssize_t[::1] outBegIdx, Py_ssize_t[::1] outNBElement, double[::1] outReal) 