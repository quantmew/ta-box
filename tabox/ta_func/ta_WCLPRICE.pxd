from .ta_utility cimport TA_INTEGER_DEFAULT
cpdef Py_ssize_t TA_WCLPRICE_Lookback()
cpdef int TA_WCLPRICE(
    Py_ssize_t startIdx,
    Py_ssize_t endIdx,
    const double[::1] inHigh,
    const double[::1] inLow,
    const double[::1] inClose,
    Py_ssize_t[::1] outBegIdx,
    Py_ssize_t[::1] outNBElement,
    double[::1] outReal
)