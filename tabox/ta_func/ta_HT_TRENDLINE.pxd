from tabox.ta_func.ta_utility cimport TA_INTEGER_DEFAULT
cdef extern from "math.h":
    double atan(double x)

from tabox.ta_func.hilbert_transform cimport HilbertVariable, do_odd, do_even

cpdef int TA_HT_TRENDLINE_Lookback()
cpdef int TA_HT_TRENDLINE(
    Py_ssize_t startIdx,
    Py_ssize_t endIdx,
    double[::1] inReal,
    Py_ssize_t[::1] outBegIdx,
    Py_ssize_t[::1] outNBElement,
    double[::1] outReal,
)