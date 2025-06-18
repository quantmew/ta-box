cdef extern from "math.h":
    cpdef double floor(double x)

cdef Py_ssize_t TA_FLOOR_Lookback()
cdef int TA_FLOOR(Py_ssize_t startIdx, Py_ssize_t endIdx, double[::1] inReal, Py_ssize_t[::1] outBegIdx, Py_ssize_t[::1] outNBElement, double[::1] outReal) 