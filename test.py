import numpy
import talib

close = numpy.random.random(100)


close = numpy.array([1,2,3,4,5], dtype=numpy.float64)
output = talib.SMA(close, timeperiod=3)
print(output)