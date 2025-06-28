from .ta_utility cimport TA_INTEGER_DEFAULT
from .ta_utility cimport TA_IS_ZERO

cdef extern from "math.h":
    double fabs(double x) nogil

cdef double TRUE_RANGE(double th, double tl, double yc) noexcept nogil
cdef double round_pos(double x) noexcept nogil