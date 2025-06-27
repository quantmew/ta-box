import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
if not cython.compiled:
    from .ta_utility import TA_INTEGER_DEFAULT
from ..settings import TA_FUNC_NO_RANGE_CHECK


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_LINEARREG_INTERCEPT_Lookback(optInTimePeriod: cython.int) -> cython.Py_ssize_t:
    """
    TA_LINEARREG_INTERCEPT_Lookback - Linear Regression Intercept Lookback

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
def TA_LINEARREG_INTERCEPT(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outReal: cython.double[::1],
) -> cython.int:
    """TA_LINEARREG_INTERCEPT - Linear Regression Intercept

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

    # 局部变量
    outIdx: cython.Py_ssize_t = 0
    today: cython.Py_ssize_t
    lookbackTotal: cython.Py_ssize_t
    SumX: cython.double
    SumXY: cython.double
    SumY: cython.double
    SumXSqr: cython.double
    Divisor: cython.double
    m: cython.double
    i: cython.Py_ssize_t
    tempValue1: cython.double

    # 计算回溯期
    lookbackTotal = TA_LINEARREG_INTERCEPT_Lookback(optInTimePeriod)

    # 调整起始索引
    if startIdx < lookbackTotal:
        startIdx = lookbackTotal

    # 检查是否有数据需要处理
    if startIdx > endIdx:
        outBegIdx[0] = 0
        outNBElement[0] = 0
        return TA_RetCode.TA_SUCCESS

    # 预计算固定值
    SumX = optInTimePeriod * (optInTimePeriod - 1) * 0.5
    SumXSqr = optInTimePeriod * (optInTimePeriod - 1) * (2 * optInTimePeriod - 1) / 6
    Divisor = SumX * SumX - optInTimePeriod * SumXSqr

    today = startIdx

    # 主计算循环
    while today <= endIdx:
        SumXY = 0.0
        SumY = 0.0
        for i in range(optInTimePeriod - 1, -1, -1):
            tempValue1 = inReal[today - i]
            SumY += tempValue1
            SumXY += cython.cast(cython.double, i) * tempValue1

        # 计算斜率 m
        m = (optInTimePeriod * SumXY - SumX * SumY) / Divisor

        # 计算截距 b 并存储到输出数组
        outReal[outIdx] = (SumY - m * SumX) / cython.cast(
            cython.double, optInTimePeriod
        )
        outIdx += 1
        today += 1

    # 设置输出参数
    outBegIdx[0] = startIdx
    outNBElement[0] = outIdx

    return TA_RetCode.TA_SUCCESS


def LINEARREG_INTERCEPT(real: np.ndarray, timeperiod: int = 14) -> np.ndarray:
    """LINEARREG_INTERCEPT(real[, timeperiod=14])

    Linear Regression Intercept (Overlap Studies)

    Linear Regression is a concept also known as the "least squares method" or "best fit."
    Linear Regression attempts to fit a straight line between several data points in such a
    way that distance between each data point and the line is minimized.

    For each point, a straight line over the specified previous bar period is determined
    in terms of y = b + m*x, where TA_LINEARREG_INTERCEPT returns 'b'.

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
    lookback: cython.Py_ssize_t = startIdx + TA_LINEARREG_INTERCEPT_Lookback(timeperiod)

    outReal = np.full_like(real, np.nan)
    outBegIdx = np.zeros(1, dtype=np.intp)
    outNBElement = np.zeros(1, dtype=np.intp)

    retCode = TA_LINEARREG_INTERCEPT(
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
