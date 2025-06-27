cpdef inline bint TA_REAL_EQ(double x, double v, double ep) nogil
cpdef inline bint TA_IS_ZERO(double v) nogil
cpdef inline bint TA_IS_ZERO_OR_NEG(double v) nogil

# Math Constants
cdef double PI

# Math Functions
cpdef double std_floor(double x)
cpdef double std_ceil(double x)
cpdef double std_fabs(double x)
cpdef double std_atan(double x)
cpdef double std_cos(double x)
cpdef double std_sin(double x)
cpdef double std_sqrt(double x)
cpdef double std_tanh(double x)
cpdef double std_tan(double x)
cpdef double std_sinh(double x)
cpdef double std_log10(double x)
cpdef double std_log(double x)
cpdef double std_exp(double x)
cpdef double std_cosh(double x)
cpdef double std_asin(double x)
cpdef double std_acos(double x)

# Helper Functions
cpdef double round_pos(double x)
cpdef double round_neg(double x)
cpdef double round_pos_2(double x)
cpdef double round_neg_2(double x)
cpdef double PER_TO_K(int per)

# K-Line Related Functions
cpdef double TA_REALBODY(double inClose, double inOpen)
cpdef double TA_UPPERSHADOW(double inHigh, double inClose, double inOpen)
cpdef double TA_LOWERSHADOW(double inClose, double inOpen, double inLow)
cpdef double TA_HIGHLOWRANGE(double inHigh, double inLow)
cpdef int TA_CANDLECOLOR(double inClose, double inOpen)
