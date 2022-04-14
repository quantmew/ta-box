
cdef extern from "math.h":
    cpdef double fabs(double x)

cdef int TA_TRANGE_Lookback()
cdef TA_TRANGE(int startIdx, int endIdx, double[::1] inHigh, double[::1] inLow, double[::1] inClose, double[::1] outReal)