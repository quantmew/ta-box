import cython
import numpy as np
from .ta_utils import check_array, check_timeperiod, check_begidx1
from ..retcode import TA_RetCode
from .ta_utility import TA_INTEGER_DEFAULT
from ..settings import TA_FUNC_NO_RANGE_CHECK
from .ta_RSI import TA_RSI, TA_RSI_Lookback
from .ta_STOCHF import TA_STOCHF, TA_STOCHF_Lookback


def TA_STOCHRSI_Lookback(
    optInTimePeriod: cython.int,
    optInFastK_Period: cython.int,
    optInFastD_Period: cython.int,
    optInFastD_MAType: cython.int,
) -> cython.Py_ssize_t:
    """
    TA_STOCHRSI_Lookback - Stochastic Relative Strength Index Lookback

    Input:
        optInTimePeriod: (int) Number of period for RSI (From 2 to 100000)
        optInFastK_Period: (int) Time period for building the Fast-K line (From 1 to 100000)
        optInFastD_Period: (int) Smoothing for making the Fast-D line (From 1 to 100000)
        optInFastD_MAType: (int) Type of Moving Average for Fast-D

    Output:
        (int) Number of lookback periods
    """
    if not TA_FUNC_NO_RANGE_CHECK:
        if optInTimePeriod == TA_INTEGER_DEFAULT:
            optInTimePeriod = 14
        elif optInTimePeriod < 2 or optInTimePeriod > 100000:
            return -1

        if optInFastK_Period == TA_INTEGER_DEFAULT:
            optInFastK_Period = 5
        elif optInFastK_Period < 1 or optInFastK_Period > 100000:
            return -1

        if optInFastD_Period == TA_INTEGER_DEFAULT:
            optInFastD_Period = 3
        elif optInFastD_Period < 1 or optInFastD_Period > 100000:
            return -1

        if optInFastD_MAType == TA_INTEGER_DEFAULT:
            optInFastD_MAType = 0
        elif optInFastD_MAType < 0 or optInFastD_MAType > 8:
            return -1

    # 计算总回溯期 = RSI回溯期 + STOCHF回溯期
    lookback_rsi = TA_RSI_Lookback(optInTimePeriod)
    lookback_stochf = TA_STOCHF_Lookback(
        optInFastK_Period, optInFastD_Period, optInFastD_MAType
    )
    return lookback_rsi + lookback_stochf


@cython.boundscheck(False)
@cython.wraparound(False)
def TA_STOCHRSI(
    startIdx: cython.Py_ssize_t,
    endIdx: cython.Py_ssize_t,
    inReal: cython.double[::1],
    optInTimePeriod: cython.int,
    optInFastK_Period: cython.int,
    optInFastD_Period: cython.int,
    optInFastD_MAType: cython.int,
    outBegIdx: cython.Py_ssize_t[::1],
    outNBElement: cython.Py_ssize_t[::1],
    outFastK: cython.double[::1],
    outFastD: cython.double[::1],
) -> cython.int:
    """
    TA_STOCHRSI - Stochastic Relative Strength Index

    Input  = double
    Output = double, double (outFastK, outFastD)

    Optional Parameters
    -------------------
    optInTimePeriod: (From 2 to 100000)
       Number of period for RSI
    optInFastK_Period: (From 1 to 100000)
       Time period for building the Fast-K line
    optInFastD_Period: (From 1 to 100000)
       Smoothing for making the Fast-D line. Usually set to 3
    optInFastD_MAType:
       Type of Moving Average for Fast-D
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

        if optInFastK_Period == TA_INTEGER_DEFAULT:
            optInFastK_Period = 5
        elif optInFastK_Period < 1 or optInFastK_Period > 100000:
            return TA_RetCode.TA_BAD_PARAM

        if optInFastD_Period == TA_INTEGER_DEFAULT:
            optInFastD_Period = 3
        elif optInFastD_Period < 1 or optInFastD_Period > 100000:
            return TA_RetCode.TA_BAD_PARAM

        if optInFastD_MAType == TA_INTEGER_DEFAULT:
            optInFastD_MAType = 0
        elif optInFastD_MAType < 0 or optInFastD_MAType > 8:
            return TA_RetCode.TA_BAD_PARAM

        if inReal is None or outFastK is None or outFastD is None:
            return TA_RetCode.TA_BAD_PARAM

    # 初始化输出参数
    outBegIdx[0] = 0
    outNBElement[0] = 0

    # 计算回溯期
    lookback_stochf = TA_STOCHF_Lookback(
        optInFastK_Period, optInFastD_Period, optInFastD_MAType
    )
    lookback_total = TA_RSI_Lookback(optInTimePeriod) + lookback_stochf

    # 调整起始索引以考虑回溯期
    if startIdx < lookback_total:
        startIdx = lookback_total

    # 检查是否有可计算的数据
    if startIdx > endIdx:
        return TA_RetCode.TA_SUCCESS

    outBegIdx[0] = startIdx

    # 计算临时数组大小
    temp_array_size = (endIdx - startIdx) + 1 + lookback_stochf
    temp_rsi_buffer = np.full(temp_array_size, np.nan, dtype=np.double)

    # 计算RSI值并存入临时缓冲区
    out_beg_idx1 = np.zeros(1, dtype=np.intp)
    out_nb_element1 = np.zeros(1, dtype=np.intp)
    ret_code = TA_RSI(
        startIdx - lookback_stochf,
        endIdx,
        inReal,
        optInTimePeriod,
        out_beg_idx1,
        out_nb_element1,
        temp_rsi_buffer,
    )

    if ret_code != TA_RetCode.TA_SUCCESS or out_nb_element1[0] == 0:
        return ret_code

    # 计算STOCHF (基于RSI值)
    ret_code = TA_STOCHF(
        0,
        temp_array_size - 1,
        temp_rsi_buffer,
        temp_rsi_buffer,
        temp_rsi_buffer,
        optInFastK_Period,
        optInFastD_Period,
        optInFastD_MAType,
        out_beg_idx1,
        outNBElement,
        outFastK,
        outFastD,
    )

    # 清理临时缓冲区
    del temp_rsi_buffer

    if ret_code != TA_RetCode.TA_SUCCESS or outNBElement[0] == 0:
        outBegIdx[0] = 0
        outNBElement[0] = 0

    return ret_code


def STOCHRSI(
    real: np.ndarray,
    timeperiod: int = 14,
    fastk_period: int = 5,
    fastd_period: int = 3,
    fastd_matype: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    STOCHRSI(real[, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0])

    Stochastic Relative Strength Index (Momentum Indicators)

    The Stochastic RSI combines the RSI with the Stochastic Oscillator
    to identify overbought/oversold conditions and potential trend reversals.

    Inputs:
        real: (any ndarray) Input series
    Parameters:
        timeperiod: 14 Number of periods for RSI calculation
        fastk_period: 5 Time period for Fast-K line
        fastd_period: 3 Smoothing period for Fast-D line
        fastd_matype: 0 Type of moving average for Fast-D (0=SMA, 1=EMA, etc.)
    Outputs:
        fastk, fastd
    """
    real = check_array(real)
    check_timeperiod(timeperiod)
    check_timeperiod(fastk_period)
    check_timeperiod(fastd_period)

    length: cython.Py_ssize_t = real.shape[0]
    start_idx: cython.Py_ssize_t = check_begidx1(real)
    end_idx: cython.Py_ssize_t = length - start_idx - 1
    lookback: cython.Py_ssize_t = start_idx + TA_STOCHRSI_Lookback(
        timeperiod, fastk_period, fastd_period, fastd_matype
    )

    out_fastk = np.full_like(real, np.nan)
    out_fastd = np.full_like(real, np.nan)
    out_beg_idx = np.zeros(1, dtype=np.intp)
    out_nb_element = np.zeros(1, dtype=np.intp)

    ret_code = TA_STOCHRSI(
        0,
        end_idx,
        real[start_idx:],
        timeperiod,
        fastk_period,
        fastd_period,
        fastd_matype,
        out_beg_idx,
        out_nb_element,
        out_fastk[lookback:],
        out_fastd[lookback:],
    )

    if ret_code != TA_RetCode.TA_SUCCESS:
        return out_fastk, out_fastd
    return out_fastk, out_fastd
