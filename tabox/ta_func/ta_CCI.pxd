cpdef Py_ssize_t TA_CCI_Lookback(int optInTimePeriod)
cpdef int TA_CCI(
    Py_ssize_t startIdx,
    Py_ssize_t endIdx,
    const double[::1] inHigh,
    const double[::1] inLow,
    const double[::1] inClose,
    int optInTimePeriod,
    Py_ssize_t[::1] outBegIdx,
    Py_ssize_t[::1] outNBElement,
    double[::1] outReal,
)