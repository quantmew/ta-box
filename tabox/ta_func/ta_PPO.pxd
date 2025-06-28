from .ta_utility cimport TA_INTEGER_DEFAULT

cdef bint TA_IS_ZERO(double v) noexcept nogil
cpdef Py_ssize_t TA_PPO_Lookback(int optInFastPeriod, int optInSlowPeriod, int optInMAType)
cpdef int TA_PPO(Py_ssize_t startIdx, Py_ssize_t endIdx, double[::1] inReal, int optInFastPeriod, int optInSlowPeriod, int optInMAType, Py_ssize_t[::1] outBegIdx, Py_ssize_t[::1] outNBElement, double[::1] outReal)
