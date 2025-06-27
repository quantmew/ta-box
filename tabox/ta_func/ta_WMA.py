import cython
import numpy as np
from .ta_utils import check_array, check_begidx1, check_timeperiod
from ..retcode import TA_RetCode


def TA_WMA_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    return optInTimePeriod - 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def TA_WMA(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1]
) -> cython.int:
    # Insert TA function code here.
    lookbackTotal: cython.Py_ssize_t = optInTimePeriod - 1

    # Move up the start index if there is not enough initial data.
    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    # Make sure there is still something to evaluate.
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    # To make the rest more efficient, handle exception case where the user is asking for a period of '1'.
    # In that case outputs equals inputs for the requested range.
    if optInTimePeriod == 1:
        outBegIdx[0] = startIdx
        outNBElement[0] = endIdx - startIdx + 1

        outReal[: outNBElement[0]] = inReal[startIdx : endIdx + 1]
        return TA_RetCode.TA_SUCCESS

    """
    Calculate the divider (always an integer value).
    By induction: 1+2+3+4+'n' = n(n+1)/2
    '>>1' is usually faster than '/2' for unsigned.
    """
    divider: cython.int = (optInTimePeriod * (optInTimePeriod + 1)) // 2

    """
    The algo used here use a very basic property of
    multiplication/addition: (x*2) = x+x
      
    As an example, a 3 period weighted can be 
    interpreted in two way:
     (x1*1)+(x2*2)+(x3*3)
         OR
     x1+x2+x2+x3+x3+x3 (this is the periodSum)
      
    When you move forward in the time serie
    you can quickly adjust the periodSum for the
    period by substracting:
      x1+x2+x3 (This is the periodSub)
    Making the new periodSum equals to:
      x2+x3+x3
    
    You can then add the new price bar
    which is x4+x4+x4 giving:
      x2+x3+x3+x4+x4+x4
    
    At this point one iteration is completed and you can
    see that we are back to the step 1 of this example.
    
    Why making it so un-intuitive? The number of memory
    access and floating point operations are kept to a
    minimum with this algo.
    """
    outIdx: cython.Py_ssize_t = 0
    trailingIdx: cython.Py_ssize_t = startIdx - lookbackTotal

    # Evaluate the initial periodSum/periodSub and trailingValue.
    periodSum: cython.double = 0.0
    periodSub: cython.double = 0.0
    inIdx: cython.Py_ssize_t = trailingIdx
    i: cython.Py_ssize_t = 1

    tempReal: cython.double = 0.0

    while inIdx < startIdx:
        tempReal = inReal[inIdx]
        inIdx += 1
        periodSub += tempReal
        periodSum += tempReal * i
        i += 1
    trailingValue: cython.double = 0.0

    # Tight loop for the requested range.
    while inIdx <= endIdx:
        # Add the current price bar to the sum who are carried through the iterations.
        tempReal = inReal[inIdx]
        inIdx += 1
        periodSub += tempReal
        periodSub -= trailingValue
        periodSum += tempReal * optInTimePeriod

        """
        /* Save the trailing value for being substract at
        * the next iteration.
        * (must be saved here just in case outReal and
        *  inReal are the same buffer).
        */
        """
        trailingValue = inReal[trailingIdx]
        trailingIdx += 1

        # Calculate the WMA for this price bar.
        outReal[outIdx] = periodSum / divider
        outIdx += 1

        # Prepare the periodSum for the next iteration.
        periodSum -= periodSub

    # Set output limits.
    outNBElement[0] = outIdx
    outBegIdx[0] = startIdx
    return TA_RetCode.TA_SUCCESS


def WMA(real: np.ndarray, timeperiod: int = 30) -> np.ndarray:
    """WMA(real[, timeperiod=?])

    Weighted Moving Average (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    real = check_array(real)
    check_timeperiod(timeperiod)

    outReal = np.full_like(real, np.nan)
    length: cython.Py_ssize_t = real.shape[0]

    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_WMA_Lookback(timeperiod)

    outBegIdx: cython.Py_ssize_t[::1] = np.zeros(shape=(1,), dtype=np.intp)
    outNBElement: cython.Py_ssize_t[::1] = np.zeros(shape=(1,), dtype=np.intp)

    TA_WMA(
        0,
        endIdx,
        real[startIdx:],
        timeperiod,
        outBegIdx,
        outNBElement,
        outReal[lookback:],
    )

    return outReal
