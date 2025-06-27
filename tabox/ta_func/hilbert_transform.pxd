from .ta_utility cimport TA_INTEGER_DEFAULT


cdef class HilbertVariable:
    cdef double odd[3]
    cdef double even[3]
    cdef double current_value
    cdef double prev_odd
    cdef double prev_even
    cdef double prev_input_odd
    cdef double prev_input_even

    cdef void do_transform(self, double input_value, int hilbert_idx, double a, double b, double adjusted_prev_period, bint is_odd)

cdef void do_odd(HilbertVariable hilbert_variable, double input_value, int hilbert_idx, double a, double b, double adjusted_prev_period)
cdef void do_even(HilbertVariable hilbert_variable, double input_value, int hilbert_idx, double a, double b, double adjusted_prev_period)