cpdef Py_ssize_t TA_OBV_Lookback()
cpdef int TA_OBV(
    Py_ssize_t startIdx,
    Py_ssize_t endIdx,
    double[::1] inReal,
    double[::1] inVolume,
    Py_ssize_t[::1] outBegIdx,
    Py_ssize_t[::1] outNBElement,
    double[::1] outReal
)