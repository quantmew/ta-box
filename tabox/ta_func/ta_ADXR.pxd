cdef extern from "math.h":
    cpdef double fabs(double x)

cdef Py_ssize_t TA_ADXR_Lookback(int optInTimePeriod)
cdef int TA_ADXR(
    Py_ssize_t startIdx,
    Py_ssize_t endIdx,
    const double[::1] inHigh,
    const double[::1] inLow,
    const double[::1] inClose,
    int optInTimePeriod,
    Py_ssize_t[::1] outBegIdx,
    Py_ssize_t[::1] outNBElement,
    double[::1] outReal
)