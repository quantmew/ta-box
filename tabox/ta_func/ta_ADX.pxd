from .ta_utility cimport TA_INTEGER_DEFAULT
from .ta_utility cimport TA_IS_ZERO

cdef extern from "math.h":
    cpdef double fabs(double x)

cdef double round_pos(double x) noexcept nogil

cdef double TRUE_RANGE(
    double th,
    double tl,
    double yc
)

cpdef Py_ssize_t TA_ADX_Lookback(int optInTimePeriod)
cpdef int TA_ADX(
    Py_ssize_t startIdx,
    Py_ssize_t endIdx,
    const double[::1] inHigh,
    const double[::1] inLow,
    const double[::1] inClose,
    int optInTimePeriod,
    Py_ssize_t[::1] outBegIdx,
    Py_ssize_t[::1] outNBElement,
    double[::1] outReal
)