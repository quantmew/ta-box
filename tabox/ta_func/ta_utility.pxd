cdef bint TA_REAL_EQ(double x, double v, double ep) noexcept nogil
cdef bint TA_IS_ZERO(double v) noexcept nogil
cdef bint TA_IS_ZERO_OR_NEG(double v) noexcept nogil

# Math Constants
cdef double PI

# Math Functions
cdef double std_floor(double x)
cdef double std_ceil(double x)
cdef double std_fabs(double x)
cdef double std_atan(double x)
cdef double std_cos(double x)
cdef double std_sin(double x)
cdef double std_sqrt(double x)
cdef double std_tanh(double x)
cdef double std_tan(double x)
cdef double std_sinh(double x)
cdef double std_log10(double x)
cdef double std_log(double x)
cdef double std_exp(double x)
cdef double std_cosh(double x)
cdef double std_asin(double x)
cdef double std_acos(double x)

# Helper Functions
cdef double round_pos(double x)
cdef double round_neg(double x)
cdef double round_pos_2(double x)
cdef double round_neg_2(double x)
cdef double PER_TO_K(int per)

# K-Line Related Functions
cpdef double TA_REALBODY(double inClose, double inOpen)
cpdef double TA_UPPERSHADOW(double inHigh, double inClose, double inOpen)
cpdef double TA_LOWERSHADOW(double inClose, double inOpen, double inLow)
cpdef double TA_HIGHLOWRANGE(double inHigh, double inLow)
cpdef int TA_CANDLECOLOR(double inClose, double inOpen)


cdef int TA_INTEGER_DEFAULT = -1