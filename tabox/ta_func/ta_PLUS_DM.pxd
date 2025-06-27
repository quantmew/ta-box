from .ta_utility cimport TA_INTEGER_DEFAULT
cpdef Py_ssize_t TA_PLUS_DM_Lookback(int optInTimePeriod)
cpdef int TA_PLUS_DM(
    Py_ssize_t startIdx,
    Py_ssize_t endIdx,
    double[::1] inHigh,
    double[::1] inLow,
    int optInTimePeriod,
    Py_ssize_t[::1] outBegIdx,
    Py_ssize_t[::1] outNBElement,
    double[::1] outReal
)