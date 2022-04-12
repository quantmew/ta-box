import numpy
import talib

close = numpy.random.random(100)


close = numpy.array([1,2,3,4,5,6,7,8,9], dtype=numpy.float64)
output = talib.SMA(close, timeperiod=3)
print(output)

from talib.tests.test_sma import test_sma

test_sma()