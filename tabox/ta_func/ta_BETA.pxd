from .ta_utility cimport TA_INTEGER_DEFAULT
from .ta_utility cimport TA_IS_ZERO

cpdef Py_ssize_t TA_BETA_Lookback(int optInTimePeriod)
cpdef int TA_BETA(
    Py_ssize_t startIdx,
    Py_ssize_t endIdx,
    double[::1] inReal0,
    double[::1] inReal1,
    int optInTimePeriod,
    Py_ssize_t[::1] outBegIdx,
    Py_ssize_t[::1] outNBElement,
    double[::1] outReal
)