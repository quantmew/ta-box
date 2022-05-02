# TA-Box

This is a Python implementation for [TA-LIB](http://ta-lib.org) based on Cython.

> TA-Lib is widely used by trading software developers requiring to perform
> technical analysis of financial market data.

> * Includes 150+ indicators such as ADX, MACD, RSI, Stochastic, Bollinger
>   Bands, etc.
> * Candlestick pattern recognition
> * Open-source API for C/C++, Java, Perl, Python and 100% Managed .NET

Because TA-Box is written in pure Python, you do not have to install the TA-Lib.
TA-Box also supports Cython compilation.
With the support of cython, the speed of TA-Box is comparable to TA-LIB.

## Installation

You can install from PyPI:

```
$ python3 -m pip install TA-Box
```

Or checkout the sources and run ``setup.py`` yourself:

```
$ python setup.py install
```

## Function List

- Cycle Indicators
  
  - HT_DCPERIOD
  
  - HT_DCPHASE
  
  - HT_PHASOR
  
  - HT_SINE
  
  - HT_TRENDMODE

- Math Operators
  
  - ADD✓
  
  - DIV✓
  
  - MAX✓
  
  - MAXINDEX
  
  - MIN✓
  
  - MININDEX
  
  - MINMAX
  
  - MINMAXINDEX
  
  - MULT✓
  
  - SUB✓
  
  - SUM✓

- Math Transform
  
  - ACOS✓
  
  - ASIN✓
  
  - ATAN✓
  
  - CEIL
  
  - COS
  
  - COSH
  
  - EXP
  
  - FLOOR
  
  - LN
  
  - LOG10
  
  - SIN✓
  
  - SINH
  
  - SQRT✓
  
  - TAN✓
  
  - TANH

- Momentum Indicators
  
  - ADX
  
  - ADXR
  
  - APO
  
  - AROON
  
  - AROONOSC
  
  - BOP
  
  - CCI
  
  - CMO
  
  - DX
  
  - MACD
  
  - MACDEXT
  
  - MACDFIX
  
  - MFI
  
  - MINUS_DI
  
  - MINUS_DM
  
  - MOM
  
  - PLUS_DI
  
  - PLUS_DM
  
  - PPO
  
  - ROC
  
  - ROCP
  
  - ROCR
  
  - ROCR100
  
  - RSI
  
  - STOCH
  
  - STOCHF
  
  - STOCHRSI
  
  - TRIX
  
  - ULTOSC
  
  - WILLR

- Overlap Studies
  
  - BBANDS
  
  - DEMA
  
  - EMA✓
  
  - HT_TRENDLINE
  
  - KAMA
  
  - MA✓
  
  - MAMA
  
  - MAVP
  
  - MIDPOINT
  
  - MIDPRICE
  
  - SAR
  
  - SAREXT
  
  - SMA✓
  
  - T3
  
  - TEMA
  
  - TRIMA
  
  - WMA

- Pattern Recognition
  
  - CDL2CROWS
  
  - CDL3BLACKCROWS
  
  - CDL3INSIDE
  
  - CDL3LINESTRIKE
  
  - CDL3OUTSIDE
  
  - CDL3STARSINSOUTH
  
  - CDL3WHITESOLDIERS
  
  - CDLABANDONEDBABY
  
  - CDLADVANCEBLOCK
  
  - CDLBELTHOLD
  
  - CDLBREAKAWAY
  
  - CDLCLOSINGMARUBOZU
  
  - CDLCONCEALBABYSWALL
  
  - CDLCOUNTERATTACK
  
  - CDLDARKCLOUDCOVER
  
  - CDLDOJI
  
  - CDLDOJISTAR
  
  - CDLDRAGONFLYDOJI
  
  - CDLENGULFING
  
  - CDLEVENINGDOJISTAR
  
  - CDLEVENINGSTAR
  
  - CDLGAPSIDESIDEWHITE
  
  - CDLGRAVESTONEDOJI
  
  - CDLHAMMER
  
  - CDLHANGINGMAN
  
  - CDLHARAMI
  
  - CDLHARAMICROSS
  
  - CDLHIGHWAVE
  
  - CDLHIKKAKE
  
  - CDLHIKKAKEMOD
  
  - CDLHOMINGPIGEON
  
  - CDLIDENTICAL3CROWS
  
  - CDLINNECK
  
  - CDLINVERTEDHAMMER
  
  - CDLKICKING
  
  - CDLKICKINGBYLENGTH
  
  - CDLLADDERBOTTOM
  
  - CDLLONGLEGGEDDOJI
  
  - CDLLONGLINE
  
  - CDLMARUBOZU
  
  - CDLMATCHINGLOW
  
  - CDLMATHOLD
  
  - CDLMORNINGDOJISTAR
  
  - CDLMORNINGSTAR
  
  - CDLONNECK
  
  - CDLPIERCING
  
  - CDLRICKSHAWMAN
  
  - CDLRISEFALL3METHODS
  
  - CDLSEPARATINGLINES
  
  - CDLSHOOTINGSTAR
  
  - CDLSHORTLINE
  
  - CDLSPINNINGTOP
  
  - CDLSTALLEDPATTERN
  
  - CDLSTICKSANDWICH
  
  - CDLTAKURI
  
  - CDLTASUKIGAP
  
  - CDLTHRUSTING
  
  - CDLTRISTAR
  
  - CDLUNIQUE3RIVER
  
  - CDLUPSIDEGAP2CROWS
  
  - CDLXSIDEGAP3METHODS

- Price Transform
  
  - AVGPRICE
  
  - MEDPRICE
  
  - TYPPRICE
  
  - WCLPRICE

- Statistic Functions
  
  - BETA
  
  - CORREL
  
  - LINEARREG
  
  - LINEARREG_ANGLE
  
  - LINEARREG_INTERCEPT
  
  - LINEARREG_SLOPE
  
  - STDDEV
  
  - TSF
  
  - VAR

- Volatility Indicators
  
  - ATR
  
  - NATR
  
  - TRANGE✓

- Volume Indicators
  
  - AD
  
  - ADOSC
  
  - OBV
