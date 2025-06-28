#!/usr/bin/env python3
"""
测试benchmark draw功能的脚本
"""

import os
import sys

def test_parse_function():
    """测试解析函数"""
    from benchmark_draw_simple import parse_benchmark_output
    
    # 模拟benchmark输出
    test_output = """
Running benchmark\bench_atan.py [11/90]
Function=bench_this_atan, TotalTime=3.7687203884124756, MaxTime=0.7690963745117188, MinTime=0.7383894920349121, MeanTime=0.7537440776824951
Function=bench_that_atan, TotalTime=3.597973585128784, MaxTime=0.8232400417327881, MinTime=0.6626050472259521, MeanTime=0.7195947170257568
Running benchmark\bench_adx.py [12/90]
Function=bench_this_adx, TotalTime=2.123456789, MaxTime=0.456789, MinTime=0.345678, MeanTime=0.4246913578
Function=bench_that_adx, TotalTime=2.987654321, MaxTime=0.567890, MinTime=0.456789, MeanTime=0.597654321
"""
    
    results = parse_benchmark_output(test_output)
    
    print("解析结果:")
    for result in results:
        print(f"函数: {result['function']}, 类型: {result['type']}, 平均时间: {result['mean_time']:.6f}")
    
    # 验证结果
    assert len(results) == 4, f"期望4个结果，实际得到{len(results)}个"
    
    # 检查atan函数
    atan_results = [r for r in results if r['function'] == 'atan']
    assert len(atan_results) == 2, "应该有两个atan结果"
    
    tabox_atan = [r for r in atan_results if r['type'] == 'tabox'][0]
    talib_atan = [r for r in atan_results if r['type'] == 'talib'][0]
    
    assert abs(tabox_atan['mean_time'] - 0.7537440776824951) < 1e-6
    assert abs(talib_atan['mean_time'] - 0.7195947170257568) < 1e-6
    
    print("解析函数测试通过！")
    return results

def test_chart_creation():
    """测试图表创建（不显示）"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        
        from benchmark_draw_simple import create_comparison_charts
        
        # 创建测试数据
        test_results = [
            {'function': 'atan', 'type': 'tabox', 'total_time': 3.77, 'max_time': 0.77, 'min_time': 0.74, 'mean_time': 0.75},
            {'function': 'atan', 'type': 'talib', 'total_time': 3.60, 'max_time': 0.82, 'min_time': 0.66, 'mean_time': 0.72},
            {'function': 'adx', 'type': 'tabox', 'total_time': 2.12, 'max_time': 0.46, 'min_time': 0.35, 'mean_time': 0.42},
            {'function': 'adx', 'type': 'talib', 'total_time': 2.99, 'max_time': 0.57, 'min_time': 0.46, 'mean_time': 0.60},
        ]
        
        # 创建图表（不显示）
        create_comparison_charts(test_results)
        
        # 检查输出文件
        assert os.path.exists('benchmark_comparison.png'), "图表文件未生成"
        assert os.path.exists('benchmark_results.csv'), "CSV文件未生成"
        
        print("图表创建测试通过！")
        
        # 清理测试文件
        if os.path.exists('benchmark_comparison.png'):
            os.remove('benchmark_comparison.png')
        if os.path.exists('benchmark_results.csv'):
            os.remove('benchmark_results.csv')
            
    except ImportError as e:
        print(f"跳过图表测试，缺少依赖: {e}")
    except Exception as e:
        print(f"图表测试失败: {e}")

def create_test_output_file():
    """创建测试输出文件"""
    test_output = """Running benchmark\bench_atan.py [11/90]
Function=bench_this_atan, TotalTime=3.7687203884124756, MaxTime=0.7690963745117188, MinTime=0.7383894920349121, MeanTime=0.7537440776824951
Function=bench_that_atan, TotalTime=3.597973585128784, MaxTime=0.8232400417327881, MinTime=0.6626050472259521, MeanTime=0.7195947170257568
Running benchmark\bench_adx.py [12/90]
Function=bench_this_adx, TotalTime=2.123456789, MaxTime=0.456789, MinTime=0.345678, MeanTime=0.4246913578
Function=bench_that_adx, TotalTime=2.987654321, MaxTime=0.567890, MinTime=0.456789, MeanTime=0.597654321
Running benchmark\bench_rsi.py [13/90]
Function=bench_this_rsi, TotalTime=1.987654321, MaxTime=0.345678, MinTime=0.234567, MeanTime=0.3975308642
Function=bench_that_rsi, TotalTime=1.876543210, MaxTime=0.345678, MinTime=0.234567, MeanTime=0.3753086420
"""
    
    with open('benchmark_output.txt', 'w', encoding='utf-8') as f:
        f.write(test_output)
    
    print("测试输出文件已创建: benchmark_output.txt")

def main():
    """主测试函数"""
    print("开始测试benchmark draw功能...")
    
    # 测试解析函数
    test_parse_function()
    
    # 测试图表创建
    test_chart_creation()
    
    # 创建测试输出文件
    create_test_output_file()
    
    print("\n所有测试完成！")
    print("\n现在可以运行:")
    print("python benchmark_draw_simple.py")

if __name__ == '__main__':
    main() 