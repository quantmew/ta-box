# define TA_REAL_EQ(x,v,ep)   (((v-ep)<x)&&(x<(v+ep)))
# define TA_IS_ZERO(v)        (((-0.00000001)<v)&&(v<0.00000001))
# define TA_IS_ZERO_OR_NEG(v) (v<0.00000001)
import cython
import math
from typing import List
from ..retcode import TA_RetCode


# 数学常量
PI = 3.14159265358979323846


def TA_REAL_EQ(x: cython.float, v: cython.float, ep: cython.float):
    return ((v - ep) < x) and (x < (v + ep))


def TA_IS_ZERO(v: cython.float):
    return ((-0.00000001) < v) and (v < 0.00000001)


def TA_IS_ZERO_OR_NEG(v: cython.float):
    return v < 0.00000001


# 数学函数
def std_floor(x: cython.float) -> cython.float:
    return math.floor(x)


def std_ceil(x: cython.float) -> cython.float:
    return math.ceil(x)


def std_fabs(x: cython.float) -> cython.float:
    return abs(x)


def std_atan(x: cython.float) -> cython.float:
    return math.atan(x)


def std_cos(x: cython.float) -> cython.float:
    return math.cos(x)


def std_sin(x: cython.float) -> cython.float:
    return math.sin(x)


def std_sqrt(x: cython.float) -> cython.float:
    return math.sqrt(x)


def std_tanh(x: cython.float) -> cython.float:
    return math.tanh(x)


def std_tan(x: cython.float) -> cython.float:
    return math.tan(x)


def std_sinh(x: cython.float) -> cython.float:
    return math.sinh(x)


def std_log10(x: cython.float) -> cython.float:
    return math.log10(x)


def std_log(x: cython.float) -> cython.float:
    return math.log(x)


def std_exp(x: cython.float) -> cython.float:
    return math.exp(x)


def std_cosh(x: cython.float) -> cython.float:
    return math.cosh(x)


def std_asin(x: cython.float) -> cython.float:
    return math.asin(x)


def std_acos(x: cython.float) -> cython.float:
    return math.acos(x)


# Utility Functions
def round_pos(x: cython.float) -> cython.float:
    return std_floor(x + 0.5)


def round_neg(x: cython.float) -> cython.float:
    return std_ceil(x - 0.5)


def round_pos_2(x: cython.float) -> cython.float:
    return (std_floor((x * 100.0) + 0.5)) / 100.0


def round_neg_2(x: cython.float) -> cython.float:
    return (std_ceil((x * 100.0) - 0.5)) / 100.0


def PER_TO_K(per: cython.int) -> cython.float:
    return 2.0 / (per + 1)


# K-Line Related Functions
def TA_REALBODY(inClose: cython.float, inOpen: cython.float) -> cython.float:
    return std_fabs(inClose - inOpen)


def TA_UPPERSHADOW(
    inHigh: cython.float, inClose: cython.float, inOpen: cython.float
) -> cython.float:
    return inHigh - (inClose if inClose >= inOpen else inOpen)


def TA_LOWERSHADOW(
    inClose: cython.float, inOpen: cython.float, inLow: cython.float
) -> cython.float:
    return (inOpen if inClose >= inOpen else inClose) - inLow


def TA_HIGHLOWRANGE(inHigh: cython.float, inLow: cython.float) -> cython.float:
    return inHigh - inLow


def TA_CANDLECOLOR(inClose: cython.float, inOpen: cython.float) -> cython.int:
    return 1 if inClose >= inOpen else -1


from enum import Enum, IntEnum

class TA_MAType(IntEnum):
    TA_MAType_SMA: cython.int = 0
    TA_MAType_EMA: cython.int = 1
    TA_MAType_WMA: cython.int = 2
    TA_MAType_DEMA: cython.int = 3
    TA_MAType_TEMA: cython.int = 4
    TA_MAType_TRIMA: cython.int = 5
    TA_MAType_KAMA: cython.int = 6
    TA_MAType_MAMA: cython.int = 7
    TA_MAType_T3: cython.int = 8

TA_INTEGER_DEFAULT: cython.int = -1


"""
/* Generated */ ENUM_BEGIN( FuncUnstId )
/* Generated */     /* 000 */  ENUM_DEFINE( TA_FUNC_UNST_ADX, Adx),
/* Generated */     /* 001 */  ENUM_DEFINE( TA_FUNC_UNST_ADXR, Adxr),
/* Generated */     /* 002 */  ENUM_DEFINE( TA_FUNC_UNST_ATR, Atr),
/* Generated */     /* 003 */  ENUM_DEFINE( TA_FUNC_UNST_CMO, Cmo),
/* Generated */     /* 004 */  ENUM_DEFINE( TA_FUNC_UNST_DX, Dx),
/* Generated */     /* 005 */  ENUM_DEFINE( TA_FUNC_UNST_EMA, Ema),
/* Generated */     /* 006 */  ENUM_DEFINE( TA_FUNC_UNST_HT_DCPERIOD, HtDcPeriod),
/* Generated */     /* 007 */  ENUM_DEFINE( TA_FUNC_UNST_HT_DCPHASE, HtDcPhase),
/* Generated */     /* 008 */  ENUM_DEFINE( TA_FUNC_UNST_HT_PHASOR, HtPhasor),
/* Generated */     /* 009 */  ENUM_DEFINE( TA_FUNC_UNST_HT_SINE, HtSine),
/* Generated */     /* 010 */  ENUM_DEFINE( TA_FUNC_UNST_HT_TRENDLINE, HtTrendline),
/* Generated */     /* 011 */  ENUM_DEFINE( TA_FUNC_UNST_HT_TRENDMODE, HtTrendMode),
/* Generated */     /* 012 */  ENUM_DEFINE( TA_FUNC_UNST_KAMA, Kama),
/* Generated */     /* 013 */  ENUM_DEFINE( TA_FUNC_UNST_MAMA, Mama),
/* Generated */     /* 014 */  ENUM_DEFINE( TA_FUNC_UNST_MFI, Mfi),
/* Generated */     /* 015 */  ENUM_DEFINE( TA_FUNC_UNST_MINUS_DI, MinusDI),
/* Generated */     /* 016 */  ENUM_DEFINE( TA_FUNC_UNST_MINUS_DM, MinusDM),
/* Generated */     /* 017 */  ENUM_DEFINE( TA_FUNC_UNST_NATR, Natr),
/* Generated */     /* 018 */  ENUM_DEFINE( TA_FUNC_UNST_PLUS_DI, PlusDI),
/* Generated */     /* 019 */  ENUM_DEFINE( TA_FUNC_UNST_PLUS_DM, PlusDM),
/* Generated */     /* 020 */  ENUM_DEFINE( TA_FUNC_UNST_RSI, Rsi),
/* Generated */     /* 021 */  ENUM_DEFINE( TA_FUNC_UNST_STOCHRSI, StochRsi),
/* Generated */     /* 022 */  ENUM_DEFINE( TA_FUNC_UNST_T3, T3),
/* Generated */                ENUM_DEFINE( TA_FUNC_UNST_ALL, FuncUnstAll),
/* Generated */                ENUM_DEFINE( TA_FUNC_UNST_NONE, FuncUnstNone) = -1
"""
class TA_FuncUnstId(IntEnum):
    TA_FUNC_UNST_ADX = 0
    TA_FUNC_UNST_ADXR = 1
    TA_FUNC_UNST_ATR = 2
    TA_FUNC_UNST_CMO = 3
    TA_FUNC_UNST_DX = 4
    TA_FUNC_UNST_EMA = 5
    TA_FUNC_UNST_HT_DCPERIOD = 6
    TA_FUNC_UNST_HT_DCPHASE = 7
    TA_FUNC_UNST_HT_PHASOR = 8
    TA_FUNC_UNST_HT_SINE = 9
    TA_FUNC_UNST_HT_TRENDLINE = 10
    TA_FUNC_UNST_HT_TRENDMODE = 11
    TA_FUNC_UNST_KAMA = 12
    TA_FUNC_UNST_MAMA = 13
    TA_FUNC_UNST_MFI = 14
    TA_FUNC_UNST_MINUS_DI = 15
    TA_FUNC_UNST_MINUS_DM = 16
    TA_FUNC_UNST_NATR = 17
    TA_FUNC_UNST_PLUS_DI = 18
    TA_FUNC_UNST_PLUS_DM = 19
    TA_FUNC_UNST_RSI = 20
    TA_FUNC_UNST_STOCHRSI = 21
    TA_FUNC_UNST_T3 = 22
    TA_FUNC_UNST_ALL = 23
    TA_FUNC_UNST_NONE = -1


class TA_Compatibility(IntEnum):
    TA_COMPATIBILITY_DEFAULT = 0
    TA_COMPATIBILITY_STANDARD = 1
    TA_COMPATIBILITY_METASTOCK = 2
    TA_COMPATIBILITY_TA_LIB = 3


class TA_Globals_t:
    unstablePeriod: List[cython.int]
    compatibility: TA_Compatibility

    def __init__(self):
        self.unstablePeriod = [0] * TA_FuncUnstId.TA_FUNC_UNST_ALL
        self.compatibility = TA_Compatibility.TA_COMPATIBILITY_DEFAULT


TA_Globals = TA_Globals_t()


def TA_GLOBALS_UNSTABLE_PERIOD(id: TA_FuncUnstId) -> cython.int:
    return TA_Globals.unstablePeriod[id]


def TA_GLOBALS_COMPATIBILITY() -> TA_Compatibility:
    return TA_Globals.compatibility


def TA_SetUnstablePeriod(id: TA_FuncUnstId, unstablePeriod: int) -> TA_RetCode:
    if id > TA_FuncUnstId.TA_FUNC_UNST_ALL:
        return TA_RetCode.TA_BAD_PARAM

    if id == TA_FuncUnstId.TA_FUNC_UNST_ALL:
        for i in range(TA_FuncUnstId.TA_FUNC_UNST_ALL):
            TA_Globals.unstablePeriod[i] = unstablePeriod
    else:
        TA_Globals.unstablePeriod[id] = unstablePeriod

    return TA_RetCode.TA_SUCCESS


def TA_GetUnstablePeriod(id: TA_FuncUnstId) -> int:
    if id >= TA_FuncUnstId.TA_FUNC_UNST_ALL:
        return 0

    return TA_Globals.unstablePeriod[id]


def TA_SetCompatibility(value: TA_Compatibility) -> TA_RetCode:
    TA_Globals.compatibility = value
    return TA_RetCode.TA_SUCCESS


def TA_GetCompatibility() -> TA_Compatibility:
    return TA_Globals.compatibility
