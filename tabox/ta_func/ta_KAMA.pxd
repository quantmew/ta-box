from .ta_utility cimport TA_INTEGER_DEFAULT
from .ta_utility cimport TA_IS_ZERO

cdef extern from "math.h":
    double fabs(double x)

cpdef Py_ssize_t TA_KAMA_Lookback(int optInTimePeriod)
cpdef int TA_KAMA(Py_ssize_t startIdx, Py_ssize_t endIdx, double[::1] inReal, int optInTimePeriod, Py_ssize_t[::1] outBegIdx, Py_ssize_t[::1] outNBElement, double[::1] outReal)
