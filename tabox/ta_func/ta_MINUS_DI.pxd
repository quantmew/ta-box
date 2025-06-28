from .ta_utility cimport TA_INTEGER_DEFAULT
from .ta_utility cimport TA_IS_ZERO

cdef extern from "math.h":
    cpdef double fabs(double x)

cpdef double TRUE_RANGE(
    double th,
    double tl,
    double yc
)

cpdef Py_ssize_t TA_MINUS_DI_Lookback(int optInTimePeriod)
cpdef int TA_MINUS_DI(
    Py_ssize_t startIdx,
    Py_ssize_t endIdx,
    double[::1] inHigh,
    double[::1] inLow,
    double[::1] inClose,
    int optInTimePeriod,
    Py_ssize_t[::1] outBegIdx,
    Py_ssize_t[::1] outNBElement,
    double[::1] outReal
)