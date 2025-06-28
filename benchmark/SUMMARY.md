# Benchmark Draw 功能总结

## 创建的文件

### 主要脚本
1. **`benchmark_draw.py`** - 完整版本，包含运行benchmark和绘图功能
2. **`benchmark_draw_simple.py`** - 简化版本，从文件读取benchmark结果并绘图
3. **`test_benchmark_draw.py`** - 测试脚本，验证功能正确性

### 配置文件
4. **`requirements_benchmark_draw.txt`** - 依赖包列表
5. **`README_benchmark_draw.md`** - 详细使用说明

### 运行脚本
6. **`run_benchmark_draw.bat`** - Windows批处理脚本
7. **`run_benchmark_draw.sh`** - Linux/macOS Shell脚本

## 功能特性

### 1. 解析功能
- 解析benchmark输出字符串
- 提取函数名、总时间、最大时间、最小时间、平均时间
- 区分TA-Box和TA-Lib函数

### 2. 图表生成
- **总执行时间比较** - 显示每个函数的总执行时间
- **平均执行时间比较** - 并排显示平均执行时间
- **性能比率分析** - 绿色表示TA-Box更快，红色表示TA-Lib更快
- **统计摘要** - 显示获胜统计和性能比率统计

### 3. 数据输出
- 控制台表格输出
- CSV文件保存
- PNG图片保存

## 使用方法

### 快速开始
```bash
# Windows
run_benchmark_draw.bat

# Linux/macOS
./run_benchmark_draw.sh
```

### 手动步骤
```bash
# 1. 安装依赖
pip install -r requirements_benchmark_draw.txt

# 2. 运行benchmark并保存输出
python benchmark_all.py > benchmark_output.txt

# 3. 生成图表
python benchmark_draw_simple.py
```

## 输出示例

### 控制台输出
```
详细性能比较:
================================================================================
函数名           TA-Box平均时间    TA-Lib平均时间    比率       更快      
================================================================================
atan            0.753744         0.719595         0.955      TA-Lib     
adx             1.234567         1.345678         1.090      TA-Box     
rsi             0.987654         0.876543         0.887      TA-Lib     
```

### 图表内容
- 4个子图，全面展示性能对比
- 颜色编码：蓝色=TA-Box，红色=TA-Lib
- 性能比率：绿色=TA-Box更快，红色=TA-Lib更快

## 技术特点

### 1. 健壮性
- 正则表达式解析，处理各种输出格式
- 错误处理和异常捕获
- 文件存在性检查

### 2. 可扩展性
- 模块化设计，易于添加新图表类型
- 支持自定义字体和样式
- 可配置的图表参数

### 3. 跨平台
- Windows批处理脚本
- Linux/macOS Shell脚本
- 中文字体自动适配

## 依赖要求

- Python 3.6+
- matplotlib >= 3.5.0
- pandas >= 1.3.0
- numpy >= 1.21.0

## 注意事项

1. **字体问题** - 如果中文显示异常，请安装中文字体
2. **路径问题** - 确保在正确的目录下运行脚本
3. **依赖问题** - 确保所有依赖包已正确安装

## 未来改进

1. 添加更多图表类型（如热力图、散点图）
2. 支持交互式图表
3. 添加性能趋势分析
4. 支持自定义benchmark输出格式
5. 添加Web界面 