from .ta_utility cimport TA_INTEGER_DEFAULT

cpdef Py_ssize_t TA_APO_Lookback(int optInFastPeriod, int optInSlowPeriod, int optInMAType)
cpdef int TA_INT_APO(
    Py_ssize_t startIdx,
    Py_ssize_t endIdx,
    const double[::1] inReal,
    int optInFastPeriod,
    int optInSlowPeriod,
    int optInMAType,
    int doPercentageOutput,
    Py_ssize_t[::1] outBegIdx,
    Py_ssize_t[::1] outNBElement,
    double[::1] outReal,
    double[::1] tempBuffer,
)
cpdef int TA_APO(
    Py_ssize_t startIdx,
    Py_ssize_t endIdx,
    const double[::1] inReal,
    int optInFastPeriod,
    int optInSlowPeriod,
    int optInMAType,
    Py_ssize_t[::1] outBegIdx,
    Py_ssize_t[::1] outNBElement,
    double[::1] outReal,
)