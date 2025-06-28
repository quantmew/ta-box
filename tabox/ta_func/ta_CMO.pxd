from .ta_utility cimport TA_INTEGER_DEFAULT
from .ta_utility cimport TA_IS_ZERO

cpdef Py_ssize_t TA_CMO_Lookback(int optInTimePeriod)
cpdef int TA_CMO(
    Py_ssize_t startIdx,
    Py_ssize_t endIdx,
    const double[::1] inReal,
    int optInTimePeriod,
    Py_ssize_t[::1] outBegIdx,
    Py_ssize_t[::1] outNBElement,
    double[::1] outReal,
)