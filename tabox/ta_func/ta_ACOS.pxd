
cdef extern from "math.h":
    cpdef double acos(double x)

cpdef Py_ssize_t TA_ACOS_Lookback()
cpdef int TA_ACOS(Py_ssize_t startIdx, Py_ssize_t endIdx, double[::1] inReal, Py_ssize_t[::1] outBegIdx, Py_ssize_t[::1] outNBElement, double[::1] outReal)