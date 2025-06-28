from .ta_utility cimport TA_INTEGER_DEFAULT
from .ta_utility cimport TA_IS_ZERO_OR_NEG

cpdef Py_ssize_t TA_BOP_Lookback()
cpdef int TA_BOP(
    Py_ssize_t startIdx,
    Py_ssize_t endIdx,
    const double[::1] inOpen,
    const double[::1] inHigh,
    const double[::1] inLow,
    const double[::1] inClose,
    Py_ssize_t[::1] outBegIdx,
    Py_ssize_t[::1] outNBElement,
    double[::1] outReal,
)