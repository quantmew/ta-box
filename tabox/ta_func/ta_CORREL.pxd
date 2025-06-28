from .ta_utility cimport TA_INTEGER_DEFAULT

cdef extern from "math.h":
    double sqrt(double x) nogil

cdef bint TA_IS_ZERO_OR_NEG(double v) noexcept nogil
cpdef Py_ssize_t TA_CORREL_Lookback(int optInTimePeriod)
cpdef int TA_CORREL(
    Py_ssize_t startIdx,
    Py_ssize_t endIdx,
    double[::1] inReal0,
    double[::1] inReal1,
    int optInTimePeriod,
    Py_ssize_t[::1] outBegIdx,
    Py_ssize_t[::1] outNBElement,
    double[::1] outReal
)