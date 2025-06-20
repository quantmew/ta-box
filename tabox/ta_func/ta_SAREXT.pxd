cpdef Py_ssize_t TA_SAREXT_Lookback(
    double optInStartValue,
    double optInOffsetOnReverse,
    double optInAccelerationInitLong,
    double optInAccelerationLong,
    double optInAccelerationMaxLong,
    double optInAccelerationInitShort,
    double optInAccelerationShort,
    double optInAccelerationMaxShort
)

cpdef int TA_SAREXT(
    Py_ssize_t startIdx,
    Py_ssize_t endIdx,
    double[::1] inHigh,
    double[::1] inLow,
    double optInStartValue,
    double optInOffsetOnReverse,
    double optInAccelerationInitLong,
    double optInAccelerationLong,
    double optInAccelerationMaxLong,
    double optInAccelerationInitShort,
    double optInAccelerationShort,
    double optInAccelerationMaxShort,
    Py_ssize_t[::1] outBegIdx,
    Py_ssize_t[::1] outNBElement,
    double[::1] outReal
)