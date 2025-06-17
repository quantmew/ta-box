cpdef bint TA_REAL_EQ(float x, float v, float ep)
cpdef bint TA_IS_ZERO(float v)
cpdef bint TA_IS_ZERO_OR_NEG(float v)

# Math Constants
cdef float PI

# Math Functions
cpdef float std_floor(float x)
cpdef float std_ceil(float x)
cpdef float std_fabs(float x)
cpdef float std_atan(float x)
cpdef float std_cos(float x)
cpdef float std_sin(float x)
cpdef float std_sqrt(float x)
cpdef float std_tanh(float x)
cpdef float std_tan(float x)
cpdef float std_sinh(float x)
cpdef float std_log10(float x)
cpdef float std_log(float x)
cpdef float std_exp(float x)
cpdef float std_cosh(float x)
cpdef float std_asin(float x)
cpdef float std_acos(float x)

# Helper Functions
cpdef float round_pos(float x)
cpdef float round_neg(float x)
cpdef float round_pos_2(float x)
cpdef float round_neg_2(float x)
cpdef float PER_TO_K(int per)

# K-Line Related Functions
cpdef float TA_REALBODY(float inClose, float inOpen)
cpdef float TA_UPPERSHADOW(float inHigh, float inClose, float inOpen)
cpdef float TA_LOWERSHADOW(float inClose, float inOpen, float inLow)
cpdef float TA_HIGHLOWRANGE(float inHigh, float inLow)
cpdef int TA_CANDLECOLOR(float inClose, float inOpen)
