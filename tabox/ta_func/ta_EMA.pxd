cdef Py_ssize_t TA_EMA_Lookback(Py_ssize_t optInTimePeriod)
cdef int TA_EMA(Py_ssize_t startIdx, Py_ssize_t endIdx, double[::1] inReal, Py_ssize_t optInTimePeriod, double[::1] outReal)