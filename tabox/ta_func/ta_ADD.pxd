from .ta_utility cimport TA_INTEGER_DEFAULT
cpdef Py_ssize_t TA_ADD_Lookback()
cpdef int TA_ADD(Py_ssize_t startIdx, Py_ssize_t endIdx, double[::1] inReal0, double[::1] inReal1, Py_ssize_t[::1] outBegIdx, Py_ssize_t[::1] outNBElement, double[::1] outReal)