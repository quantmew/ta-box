cpdef Py_ssize_t TA_MAVP_Lookback(int optInMinPeriod, int optInMaxPeriod, int optInMAType)
cpdef int TA_MAVP(
    Py_ssize_t startIdx,
    Py_ssize_t endIdx,
    double[::1] inReal,
    double[::1] inPeriods,
    int optInMinPeriod,
    int optInMaxPeriod,
    int optInMAType,
    Py_ssize_t[::1] outBegIdx,
    Py_ssize_t[::1] outNBElement,
    double[::1] outReal
)
