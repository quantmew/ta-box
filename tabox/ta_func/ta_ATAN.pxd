
cdef extern from "math.h":
    cpdef double atan(double x)

cdef int TA_ATAN_Lookback()
cdef TA_ATAN(int startIdx, int endIdx, double[::1] inReal, double[::1] outReal)