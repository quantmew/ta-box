from .ta_utility cimport TA_INTEGER_DEFAULT
cpdef Py_ssize_t TA_AROON_Lookback(int optInTimePeriod)
cpdef int TA_INT_AROON(
    Py_ssize_t startIdx,
    Py_ssize_t endIdx,
    const double[::1] inHigh,
    const double[::1] inLow,
    int optInTimePeriod,
    Py_ssize_t[::1] outBegIdx,
    Py_ssize_t[::1] outNBElement,
    double[::1] outAroonDown,
    double[::1] outAroonUp,
)

cpdef int TA_AROON(
    Py_ssize_t startIdx,
    Py_ssize_t endIdx,
    const double[::1] inHigh,
    const double[::1] inLow,
    int optInTimePeriod,
    Py_ssize_t[::1] outBegIdx,
    Py_ssize_t[::1] outNBElement,
    double[::1] outAroonDown,
    double[::1] outAroonUp,
)