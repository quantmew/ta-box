from .ta_utility cimport TA_INTEGER_DEFAULT
cpdef Py_ssize_t TA_MA_Lookback(int optInTimePeriod, int optInMAType)
cpdef int TA_MA(Py_ssize_t startIdx, Py_ssize_t endIdx, double[::1] inReal, int optInTimePeriod, int optInMAType, Py_ssize_t[::1] outBegIdx, Py_ssize_t[::1] outNBElement, double[::1] outReal) 