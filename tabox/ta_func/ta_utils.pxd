from .ta_utility cimport TA_INTEGER_DEFAULT
cpdef void check_timeperiod(int timeperiod)
cpdef int check_begidx1(double[::1] a1)
cpdef int check_begidx2(double[::1] a1, double[::1] a2)
cpdef int check_begidx3(double[::1] a1, double[::1] a2, double[::1] a3)
cpdef int check_begidx4(double[::1] a1, double[::1] a2, double[::1] a3, double[::1] a4)
cpdef check_length2(double[::1] a1, double[::1] a2)
cpdef check_length3(double[::1] a1, double[::1] a2, double[::1] a3)
cpdef check_length4(double[::1] a1, double[::1] a2, double[::1] a3, double[::1] a4)