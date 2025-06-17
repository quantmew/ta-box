import cython

cpdef Py_ssize_t TA_MININDEX_Lookback(Py_ssize_t optInTimePeriod) noexcept nogil

cpdef int TA_MININDEX(
    Py_ssize_t startIdx,
    Py_ssize_t endIdx,
    double[::1] inReal,
    int optInTimePeriod,
    Py_ssize_t[::1] outBegIdx,
    Py_ssize_t[::1] outNBElement,
    Py_ssize_t[::1] outInteger,
)