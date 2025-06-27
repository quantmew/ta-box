from .ta_utility cimport TA_INTEGER_DEFAULT
cpdef Py_ssize_t TA_DIV_Lookback()
cpdef int TA_DIV(
    Py_ssize_t startIdx,
    Py_ssize_t endIdx,
    const double[::1] inReal0,
    const double[::1] inReal1,
    Py_ssize_t[::1] outBegIdx,
    Py_ssize_t[::1] outNBElement,
    double[::1] outReal,
)