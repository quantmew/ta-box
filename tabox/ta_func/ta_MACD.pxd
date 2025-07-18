from .ta_utility cimport TA_INTEGER_DEFAULT
from .ta_utility cimport PER_TO_K

cpdef Py_ssize_t TA_MACD_Lookback(int optInFastPeriod, int optInSlowPeriod, int optInSignalPeriod)
cpdef int TA_INT_MACD(Py_ssize_t startIdx, Py_ssize_t endIdx, double[::1] inReal, int optInFastPeriod, int optInSlowPeriod, int optInSignalPeriod_2, Py_ssize_t[::1] outBegIdx, Py_ssize_t[::1] outNBElement, double[::1] outMACD, double[::1] outMACDSignal, double[::1] outMACDHist)
cpdef int TA_MACD(Py_ssize_t startIdx, Py_ssize_t endIdx, double[::1] inReal, int optInFastPeriod, int optInSlowPeriod, int optInSignalPeriod, Py_ssize_t[::1] outBegIdx, Py_ssize_t[::1] outNBElement, double[::1] outMACD, double[::1] outMACDSignal, double[::1] outMACDHist)