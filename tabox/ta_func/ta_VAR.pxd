from .ta_utility cimport TA_INTEGER_DEFAULT
cpdef Py_ssize_t TA_VAR_Lookback(int optInTimePeriod, double optInNbDev)
cpdef int TA_INT_VAR(Py_ssize_t startIdx, Py_ssize_t endIdx, double[::1] inReal, int optInTimePeriod, Py_ssize_t[::1] outBegIdx, Py_ssize_t[::1] outNBElement, double[::1] outReal)
cpdef int TA_VAR(Py_ssize_t startIdx, Py_ssize_t endIdx, double[::1] inReal, int optInTimePeriod, double optInNbDev, Py_ssize_t[::1] outBegIdx, Py_ssize_t[::1] outNBElement, double[::1] outReal) 