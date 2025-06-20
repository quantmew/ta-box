cpdef Py_ssize_t TA_SAR_Lookback(double optInAcceleration, double optInMaximum)
cpdef int TA_SAR(
    Py_ssize_t startIdx,
    Py_ssize_t endIdx,
    double[::1] inHigh,
    double[::1] inLow,
    double optInAcceleration,
    double optInMaximum,
    Py_ssize_t[::1] outBegIdx,
    Py_ssize_t[::1] outNBElement,
    double[::1] outReal
)