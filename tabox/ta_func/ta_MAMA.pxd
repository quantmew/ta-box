from .ta_utility cimport TA_INTEGER_DEFAULT
cdef extern from "math.h":
    double atan(double x)

from tabox.ta_func.hilbert_transform cimport HilbertVariable, do_odd, do_even
cpdef Py_ssize_t TA_MAMA_Lookback(double optInFastLimit, double optInSlowLimit)
cpdef int TA_MAMA(Py_ssize_t startIdx, Py_ssize_t endIdx, double[::1] inReal, double optInFastLimit, double optInSlowLimit, Py_ssize_t[::1] outBegIdx, Py_ssize_t[::1] outNBElement, double[::1] outMAMA, double[::1] outFAMA)
