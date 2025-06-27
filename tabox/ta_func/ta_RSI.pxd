cdef inline bint TA_IS_ZERO(double v) nogil

cpdef Py_ssize_t TA_RSI_Lookback(int optInTimePeriod)
cpdef int TA_RSI(Py_ssize_t startIdx, Py_ssize_t endIdx, double[::1] inReal, int optInTimePeriod, Py_ssize_t[::1] outBegIdx, Py_ssize_t[::1] outNBElement, double[::1] outReal)