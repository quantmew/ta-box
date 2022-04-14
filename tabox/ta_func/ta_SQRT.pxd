
cdef extern from "math.h":
    cpdef double sqrt(double x)

cdef int TA_SQRT_Lookback()
cdef TA_SQRT(int startIdx, int endIdx, double[::1] inReal, double[::1] outReal)