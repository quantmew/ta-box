cdef extern from "math.h":
    cpdef double sqrt(double x)

cpdef Py_ssize_t TA_STDDEV_Lookback(int optInTimePeriod)
cpdef int TA_STDDEV(Py_ssize_t startIdx, Py_ssize_t endIdx, double[::1] inReal, int optInTimePeriod, double optInNbDev, Py_ssize_t[::1] outBegIdx, Py_ssize_t[::1] outNBElement, double[::1] outReal) 