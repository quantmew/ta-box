cdef extern from "math.h":
    cpdef double atan(double x)

cpdef Py_ssize_t TA_LINEARREG_ANGLE_Lookback(int optInTimePeriod)
cpdef int TA_LINEARREG_ANGLE(Py_ssize_t startIdx, Py_ssize_t endIdx, double[::1] inReal, int optInTimePeriod, Py_ssize_t[::1] outBegIdx, Py_ssize_t[::1] outNBElement, double[::1] outReal)