cpdef Py_ssize_t TA_NATR_Lookback(int optInTimePeriod)
cpdef int TA_NATR(
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