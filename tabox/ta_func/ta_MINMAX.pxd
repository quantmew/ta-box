from .ta_utility cimport TA_INTEGER_DEFAULT
import cython

cpdef Py_ssize_t TA_MINMAX_Lookback(Py_ssize_t optInTimePeriod) noexcept nogil

cpdef int TA_MINMAX(
    Py_ssize_t startIdx,
    Py_ssize_t endIdx,
    double[::1] inReal,
    int optInTimePeriod,
    Py_ssize_t[::1] outBegIdx,
    Py_ssize_t[::1] outNBElement,
    double[::1] outMin,
    double[::1] outMax,
)