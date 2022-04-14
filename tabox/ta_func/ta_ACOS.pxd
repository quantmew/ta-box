
cdef extern from "math.h":
    cpdef double acos(double x)

cdef int TA_ACOS_Lookback()
cdef TA_ACOS(int startIdx, int endIdx, double[::1] inReal, double[::1] outReal)