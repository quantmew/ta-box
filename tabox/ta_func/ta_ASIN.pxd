
cdef extern from "math.h":
    cpdef double asin(double x)

cdef int TA_ASIN_Lookback()
cdef TA_ASIN(int startIdx, int endIdx, double[::1] inReal, double[::1] outReal)