# Benchmark Draw 使用说明

这个工具用于在完成所有benchmark后绘制比较图表，直观地展示TA-Box和TA-Lib的性能差异。

## 功能特性

- 解析benchmark输出结果
- 生成4个比较图表：
  1. 总执行时间比较
  2. 平均执行时间比较  
  3. 性能比率分析
  4. 统计摘要
- 输出详细的性能比较表格
- 保存结果到CSV文件和PNG图片

## 安装依赖

```bash
pip install -r requirements_benchmark_draw.txt
```

或者手动安装：

```bash
pip install matplotlib pandas numpy
```

## 使用方法

### 方法1：完整流程（推荐）

1. 运行所有benchmark并保存输出：
```bash
cd benchmark
python benchmark_all.py > benchmark_output.txt
```

2. 生成比较图表：
```bash
python benchmark_draw_simple.py
```

### 方法2：直接运行（包含benchmark执行）

```bash
cd benchmark
python benchmark_draw.py
```

## 输出文件

- `benchmark_comparison.png` - 比较图表
- `benchmark_results.csv` - 原始数据
- `benchmark_output.txt` - benchmark输出（方法1）

## 图表说明

### 1. 总执行时间比较
显示每个函数的总执行时间，蓝色表示TA-Box，红色表示TA-Lib。

### 2. 平均执行时间比较
显示每个函数的平均执行时间，便于比较性能差异。

### 3. 性能比率分析
- 绿色柱：TA-Box更快（比率 > 1）
- 红色柱：TA-Lib更快（比率 < 1）
- 虚线：性能相等线（比率 = 1）

### 4. 统计摘要
- 总函数数量
- 各库获胜的函数数量和百分比
- 平均、最大、最小性能比率

## 控制台输出示例

```
详细性能比较:
================================================================================
函数名           TA-Box平均时间    TA-Lib平均时间    比率       更快      
================================================================================
atan            0.753744         0.719595         0.955      TA-Lib     
adx             1.234567         1.345678         1.090      TA-Box     
rsi             0.987654         0.876543         0.887      TA-Lib     
...
```

## 注意事项

1. 确保benchmark输出格式正确，包含以下信息：
   - Function=函数名
   - TotalTime=总时间
   - MaxTime=最大时间
   - MinTime=最小时间
   - MeanTime=平均时间

2. 函数名必须以`bench_this_`或`bench_that_`开头

3. 如果遇到中文显示问题，请确保系统安装了中文字体

## 故障排除

### 中文显示问题
如果图表中的中文显示为方块，请尝试：

1. 安装中文字体：
   - Windows: 安装SimHei或Microsoft YaHei
   - Linux: `sudo apt-get install fonts-wqy-zenhei`
   - macOS: 系统自带中文字体

2. 修改字体设置：
```python
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
```

### 依赖安装问题
如果安装matplotlib或pandas失败，请尝试：

```bash
pip install --upgrade pip
pip install matplotlib pandas numpy
```

### 文件路径问题
确保在正确的目录下运行脚本，或者修改文件路径。 