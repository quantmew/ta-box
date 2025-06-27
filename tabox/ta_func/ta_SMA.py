import cython
import numpy as np
from .ta_utils import check_array, check_begidx1, check_timeperiod
from ..retcode import TA_RetCode
from ..settings import TA_FUNC_NO_RANGE_CHECK
from .ta_utility import TA_GLOBALS_UNSTABLE_PERIOD

if not cython.compiled:
    from .ta_utility import TA_INTEGER_DEFAULT

def TA_SMA_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """
    Calculate the lookback period required for SMA
    """
    # Check parameter range
    if not TA_FUNC_NO_RANGE_CHECK:
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 30
        elif optInTimePeriod < 2 or optInTimePeriod > 100000:
            return -1
    
    return optInTimePeriod - 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def TA_SMA(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """
    Calculate Simple Moving Average (SMA)
    
    Parameters:
        startIdx: Starting index
        endIdx: Ending index
        inReal: Input data array
        optInTimePeriod: Time period
        outBegIdx: Output starting index
        outNBElement: Number of output elements
        outReal: Output result array
    """
    # Range check
    if not TA_FUNC_NO_RANGE_CHECK:
        # Validate starting index
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        
        # Validate ending index
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        
        # Validate input array
        if inReal is None:
            return TA_RetCode.TA_BAD_PARAM
        
        # Validate time period
        if optInTimePeriod == -1:  # TA_INTEGER_DEFAULT
            optInTimePeriod = 30
        elif optInTimePeriod < 2 or optInTimePeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM
        
        # Validate output array
        if outReal is None:
            return TA_RetCode.TA_BAD_PARAM
    
    # Calculate lookback period
    lookbackTotal: cython.Py_ssize_t = optInTimePeriod - 1
    
    # Adjust starting index
    if startIdx < lookbackTotal:
        startIdx = lookbackTotal
    
    # Check if there is data to process
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS
    
    # Initialize cumulative sum
    periodTotal: cython.double = 0.0
    trailingIdx: cython.Py_ssize_t = startIdx - lookbackTotal
    
    # Sum values for the initial period
    i: cython.Py_ssize_t = trailingIdx
    if optInTimePeriod > 1:
        while i < startIdx:
            periodTotal += inReal[i]
            i += 1
    
    # Calculate SMA
    outIdx: cython.Py_ssize_t = 0
    while True:
        periodTotal += inReal[i]
        i += 1
        
        tempReal: cython.double = periodTotal
        
        periodTotal -= inReal[trailingIdx]
        trailingIdx += 1
        
        outReal[outIdx] = tempReal / optInTimePeriod
        outIdx += 1
        
        if i > endIdx:
            break
    
    # Set output parameters
    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx
    
    return TA_RetCode.TA_SUCCESS


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def TA_INT_SMA(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """
    Integer version of SMA calculation
    
    Parameters:
        startIdx: Starting index
        endIdx: Ending index
        inReal: Input data array
        optInTimePeriod: Time period
        outBegIdx: Output starting index
        outNBElement: Number of output elements
        outReal: Output result array
    """
    # Range check
    if not TA_FUNC_NO_RANGE_CHECK:
        # Validate starting index
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        
        # Validate ending index
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        
        # Validate input array
        if inReal is None:
            return TA_RetCode.TA_BAD_PARAM
        
        # Validate time period
        if optInTimePeriod == -1:  # TA_INTEGER_DEFAULT
            optInTimePeriod = 30
        elif optInTimePeriod < 2 or optInTimePeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM
        
        # Validate output array
        if outReal is None:
            return TA_RetCode.TA_BAD_PARAM
    
    # Calculate lookback period
    lookbackTotal: cython.int = optInTimePeriod - 1
    
    # Adjust starting index
    if startIdx < lookbackTotal:
        startIdx = lookbackTotal
    
    # Check if there is data to process
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS
    
    # Initialize cumulative sum
    periodTotal: cython.double = 0.0  # Changed to double to match C implementation
    trailingIdx: cython.int = startIdx - lookbackTotal
    
    # Sum values for the initial period
    i: cython.int = trailingIdx
    if optInTimePeriod > 1:
        while i < startIdx:
            periodTotal += inReal[i]
            i += 1
    
    # Calculate SMA
    outIdx: cython.int = 0
    while True:
        periodTotal += inReal[i]
        i += 1
        
        tempReal: cython.double = periodTotal
        
        periodTotal -= inReal[trailingIdx]
        trailingIdx += 1
        
        outReal[outIdx] = tempReal / optInTimePeriod
        outIdx += 1
        
        if i > endIdx:
            break
    
    # Set output parameters
    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx
    
    return TA_RetCode.TA_SUCCESS


def SMA(real: np.ndarray, timeperiod: int = 30) -> np.ndarray:
    """SMA(real[, timeperiod=?])

    Simple Moving Average (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    # Check input array
    real = check_array(real)
    
    # Check time period
    if not TA_FUNC_NO_RANGE_CHECK:
        check_timeperiod(timeperiod)
    
    # Create output array
    outReal = np.full_like(real, np.nan)
    length: cython.Py_ssize_t = real.shape[0]
    
    # Calculate starting index
    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_SMA_Lookback(timeperiod)
    
    # Check if lookback period is valid
    if lookback < 0:
        raise ValueError("Invalid timeperiod")
    
    # Initialize output parameters
    outBegIdx: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    outNBElement: cython.Py_ssize_t[::1] = np.zeros(1, dtype=np.intp)
    
    # Call core calculation function
    retcode = TA_SMA(0, endIdx, real[startIdx:], timeperiod, outBegIdx, outNBElement, outReal[lookback:])
    
    # Check return code
    if retcode != TA_RetCode.TA_SUCCESS:
        raise RuntimeError(f"TA_SMA failed with error code: {retcode}")
    
    return outReal