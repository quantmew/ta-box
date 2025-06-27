import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
from .ta_utility import TA_INTEGER_DEFAULT
from ..settings import TA_FUNC_NO_RANGE_CHECK


def TA_LINEARREG_SLOPE_Lookback(
    optInTimePeriod: cython.int,
) -> cython.Py_ssize_t:
    """
    TA_LINEARREG_SLOPE_Lookback - Linear Regression Slope Lookback

    Input:
        optInTimePeriod: (int) Number of period (From 2 to 100000)

    Output:
        (int) Number of lookback periods
    """
    if not TA_FUNC_NO_RANGE_CHECK:
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 14
        elif optInTimePeriod < 2 or optInTimePeriod > 100000:
            return -1
    return optInTimePeriod - 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def TA_LINEARREG_SLOPE(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """
    TA_LINEARREG_SLOPE - Linear Regression Slope

    Input  = double
    Output = double

    Optional Parameters
    -------------------
    optInTimePeriod:(From 2 to 100000)
       Number of period
    """
    # 参数检查
    if not TA_FUNC_NO_RANGE_CHECK:
        if startIdx < 0:
            return TA_RetCode.TA_OUT_OF_RANGE_START_INDEX
        if endIdx < 0 or endIdx < startIdx:
            return TA_RetCode.TA_OUT_OF_RANGE_END_INDEX
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 14
        elif optInTimePeriod < 2 or optInTimePeriod > 100000:
            return TA_RetCode.TA_BAD_PARAM
        if inReal is None or outReal is None:
            return TA_RetCode.TA_BAD_PARAM

    """
    Linear Regression is a concept also known as the
    "least squares method" or "best fit." Linear
    Regression attempts to fit a straight line between
    several data points in such a way that distance
    between each data point and the line is minimized.

    For each point, a straight line over the specified
    previous bar period is determined in terms
    of y = b + m*x:

    TA_LINEARREG          : Returns b+m*(period-1)
    TA_LINEARREG_SLOPE    : Returns 'm'
    TA_LINEARREG_ANGLE    : Returns 'm' in degree.
    TA_LINEARREG_INTERCEPT: Returns 'b'
    TA_TSF                : Returns b+m*(period)
    """

    # 调整startIdx以考虑回溯期
    lookbackTotal: cython.Py_ssize_t = TA_LINEARREG_SLOPE_Lookback(optInTimePeriod)
    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    # 确保仍有数据可评估
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    outIdx: cython.Py_ssize_t = 0  # 输出索引
    today: cython.Py_ssize_t = startIdx

    # 预计算固定值
    SumX: cython.double = optInTimePeriod * (optInTimePeriod - 1) * 0.5
    SumXSqr: cython.double = optInTimePeriod * (optInTimePeriod - 1) * (2 * optInTimePeriod - 1) / 6
    Divisor: cython.double = SumX * SumX - optInTimePeriod * SumXSqr

    SumXY: cython.double = 0.0
    SumY: cython.double = 0.0
    tempValue1: cython.double = 0.0
    i: cython.Py_ssize_t = 0

    # 处理除零情况
    if Divisor == 0:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    while today <= endIdx:
        SumXY = 0.0
        SumY = 0.0
        for i in range(optInTimePeriod - 1, -1, -1):
            tempValue1 = inReal[today - i]
            SumY += tempValue1
            SumXY += cython.cast(cython.double, i) * tempValue1

        outReal[outIdx] = (optInTimePeriod * SumXY - SumX * SumY) / Divisor
        outIdx += 1
        today += 1

    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx
    return TA_RetCode.TA_SUCCESS


def LINEARREG_SLOPE(real: np.ndarray, timeperiod: int = 14) -> np.ndarray:
    """
    LINEARREG_SLOPE(real[, timeperiod=14])

    Linear Regression Slope (Overlap Studies)

    The linear regression slope is a measure of the trend strength and direction
    over a specified period using linear regression analysis.

    Inputs:
        real: (any ndarray) Input series
    Parameters:
        timeperiod: 14 Number of periods
    Outputs:
        real
    """
    real = check_array(real)
    check_timeperiod(timeperiod)

    length: cython.Py_ssize_t = real.shape[0]
    startIdx: cython.Py_ssize_t = check_begidx1(real)
    endIdx: cython.Py_ssize_t = length - startIdx - 1
    lookback: cython.Py_ssize_t = startIdx + TA_LINEARREG_SLOPE_Lookback(timeperiod)

    outReal = np.full_like(real, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_LINEARREG_SLOPE(
        0,
        endIdx,
        real[startIdx:],
        timeperiod,
        outBegIdx,
        outNBElement,
        outReal[lookback:],
    )
    if retCode != TA_RetCode.TA_SUCCESS:
        return outReal
    return outReal
