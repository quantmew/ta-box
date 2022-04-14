# TA-Box

这是基于 Cython 的 [TA-LIB](http://ta-lib.org) 的 Python 实现。

> TA-Lib 被需要执行的交易软件开发人员广泛使用
> 金融市场数据的技术分析。

> * 包括 150 多种指标，如 ADX、MACD、RSI、随机指标、布林线等
> * K线模式识别
> * 用于 C/C++、Java、Perl、Python 和 100% 托管 .NET 的开源 API

因为 TA-Box 是用纯 Python 编写的，所以您不必安装 TA-Lib。
TA-Box 还支持 Cython 编译。
在cython的支持下，TA-Box的速度堪比TA-LIB。

## 安装

您可以从 PyPI 安装：

```
$ python3 -m pip install TA-Box
```

或者下载源代码并运行“setup.py”：

```
$ python setup.py install
```