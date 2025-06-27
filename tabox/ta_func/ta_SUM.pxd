from .ta_utility cimport TA_INTEGER_DEFAULT
cpdef Py_ssize_t TA_SUM_Lookback(int optInTimePeriod)
cpdef int TA_SUM(Py_ssize_t startIdx, Py_ssize_t endIdx, double[::1] inReal, int optInTimePeriod, double[::1] outReal)