from .ta_utility cimport TA_INTEGER_DEFAULT
cpdef Py_ssize_t TA_T3_Lookback(int optInTimePeriod, double optInVFactor)
cpdef int TA_T3(Py_ssize_t startIdx, Py_ssize_t endIdx, double[::1] inReal, int optInTimePeriod, double optInVFactor, Py_ssize_t[::1] outBegIdx, Py_ssize_t[::1] outNBElement, double[::1] outReal) 