from .ta_utility cimport TA_INTEGER_DEFAULT
cpdef Py_ssize_t TA_AD_Lookback()
cpdef int TA_AD(Py_ssize_t startIdx, Py_ssize_t endIdx, double[::1] inHigh, double[::1] inLow, double[::1] inClose, double[::1] inVolume, Py_ssize_t[::1] outBegIdx, Py_ssize_t[::1] outNBElement, double[::1] outReal)