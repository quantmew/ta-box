from .ta_utility cimport TA_INTEGER_DEFAULT
cpdef int TA_TSF_Lookback(int optInTimePeriod)
cpdef int TA_TSF(Py_ssize_t startIdx, Py_ssize_t endIdx, double[::1] inReal, int optInTimePeriod, Py_ssize_t[::1] outBegIdx, Py_ssize_t[::1] outNBElement, double[::1] outReal)