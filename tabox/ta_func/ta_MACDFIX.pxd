from .ta_utility cimport TA_INTEGER_DEFAULT
cpdef Py_ssize_t TA_MACDFIX_Lookback(int optInSignalPeriod)
cpdef int TA_MACDFIX(Py_ssize_t startIdx, Py_ssize_t endIdx, double[::1] inReal, int optInSignalPeriod, Py_ssize_t[::1] outBegIdx, Py_ssize_t[::1] outNBElement, double[::1] outMACD, double[::1] outMACDSignal, double[::1] outMACDHist)