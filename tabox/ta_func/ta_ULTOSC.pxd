from .ta_utility cimport TA_INTEGER_DEFAULT
cdef extern from "math.h":
    double fabs(double x)
    double fmin(double x, double y)


cdef bint TA_IS_ZERO(double v) noexcept nogil

cdef extern from *:
    """
    #define CALC_TERMS(day, trueLow, trueRange, closeMinusTrueLow, inLow, inHigh, inClose)         \
    {                                                                  \
        double tempLT = ((double *) inLow.data)[day];                  \
        double tempHT = ((double *) inHigh.data)[day];                 \
        double tempCY = ((double *) inClose.data)[day-1];              \
        trueLow = fmin( tempLT, tempCY );                               \
        closeMinusTrueLow = ((double *) inClose.data)[day] - trueLow;  \
        trueRange = tempHT - tempLT;                                   \
        double tempDouble = fabs( tempCY - tempHT );                   \
        if( tempDouble > trueRange )                                   \
            trueRange = tempDouble;                                    \
        tempDouble = fabs( tempCY - tempLT  );                         \
        if( tempDouble > trueRange )                                   \
            trueRange = tempDouble;                                    \
    }
    """
    void CALC_TERMS(int day, double trueLow, double trueRange, double closeMinusTrueLow, double[::1] inLow, double[::1] inHigh, double[::1] inClose)

cdef tuple[double, double, double] calc_terms(Py_ssize_t day, double[::1] inLow, double[::1] inHigh, double[::1] inClose)
cdef tuple[double, double] prime_totals(int period, Py_ssize_t startIdx, double[::1] inLow, double[::1] inHigh, double[::1] inClose)
cpdef Py_ssize_t TA_ULTOSC_Lookback(int optInTimePeriod1, int optInTimePeriod2, int optInTimePeriod3)
cpdef int TA_ULTOSC(Py_ssize_t startIdx, Py_ssize_t endIdx, double[::1] inHigh, double[::1] inLow, double[::1] inClose, int optInTimePeriod1, int optInTimePeriod2, int optInTimePeriod3, Py_ssize_t[::1] outBegIdx, Py_ssize_t[::1] outNBElement, double[::1] outReal)