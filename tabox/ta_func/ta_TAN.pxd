
cdef extern from "math.h":
    cpdef double tan(double x)

cdef int TA_TAN_Lookback()
cdef TA_TAN(int startIdx, int endIdx, double[::1] inReal, double[::1] outReal)