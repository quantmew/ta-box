cpdef Py_ssize_t TA_AROONOSC_Lookback(int optInTimePeriod)
cpdef int TA_INT_AROONOSC(
    Py_ssize_t startIdx,
    Py_ssize_t endIdx,
    const double[::1] inHigh,
    const double[::1] inLow,
    int optInTimePeriod,
    Py_ssize_t[::1] outBegIdx,
    Py_ssize_t[::1] outNBElement,
    double[::1] outReal,
)
cpdef int TA_AROONOSC(
    Py_ssize_t startIdx,
    Py_ssize_t endIdx,
    const double[::1] inHigh,
    const double[::1] inLow,
    int optInTimePeriod,
    Py_ssize_t[::1] outBegIdx,
    Py_ssize_t[::1] outNBElement,
    double[::1] outReal,
)