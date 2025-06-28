import glob
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_benchmark_output(output_text):
    """
    解析benchmark输出文本，提取函数名和性能数据
    """
    results = []
    
    # 匹配benchmark输出格式的正则表达式
    pattern = r'Function=([^,]+), TotalTime=([\d.]+), MaxTime=([\d.]+), MinTime=([\d.]+), MeanTime=([\d.]+)'
    
    for line in output_text.split('\n'):
        match = re.search(pattern, line)
        if match:
            func_name = match.group(1)
            total_time = float(match.group(2))
            max_time = float(match.group(3))
            min_time = float(match.group(4))
            mean_time = float(match.group(5))
            
            # 提取函数类型（this_ 或 that_）
            if func_name.startswith('bench_this_'):
                func_type = 'tabox'
                func_short = func_name.replace('bench_this_', '')
            elif func_name.startswith('bench_that_'):
                func_type = 'talib'
                func_short = func_name.replace('bench_that_', '')
            else:
                continue
            
            results.append({
                'function': func_short,
                'type': func_type,
                'total_time': total_time,
                'max_time': max_time,
                'min_time': min_time,
                'mean_time': mean_time
            })
    
    return results

def create_comparison_charts(results):
    """
    创建比较图表
    """
    if not results:
        print("没有找到benchmark结果")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 按函数名分组
    functions = df['function'].unique()
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('TA-Box vs TA-Lib 性能比较', fontsize=16, fontweight='bold')
    
    # 1. 总时间比较
    ax1 = axes[0, 0]
    for func in functions:
        tabox_data = df[(df['function'] == func) & (df['type'] == 'tabox')]
        talib_data = df[(df['function'] == func) & (df['type'] == 'talib')]
        
        if not tabox_data.empty and not talib_data.empty:
            x = [func]
            tabox_time = tabox_data['total_time'].iloc[0]
            talib_time = talib_data['total_time'].iloc[0]
            
            ax1.bar([f"{func}_tabox"], [tabox_time], color='skyblue', alpha=0.7, label='TA-Box' if func == functions[0] else "")
            ax1.bar([f"{func}_talib"], [talib_time], color='lightcoral', alpha=0.7, label='TA-Lib' if func == functions[0] else "")
    
    ax1.set_title('总执行时间比较')
    ax1.set_ylabel('时间 (秒)')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. 平均时间比较
    ax2 = axes[0, 1]
    tabox_means = []
    talib_means = []
    func_labels = []
    
    for func in functions:
        tabox_data = df[(df['function'] == func) & (df['type'] == 'tabox')]
        talib_data = df[(df['function'] == func) & (df['type'] == 'talib')]
        
        if not tabox_data.empty and not talib_data.empty:
            tabox_means.append(tabox_data['mean_time'].iloc[0])
            talib_means.append(talib_data['mean_time'].iloc[0])
            func_labels.append(func)
    
    x = np.arange(len(func_labels))
    width = 0.35
    
    ax2.bar(x - width/2, tabox_means, width, label='TA-Box', color='skyblue', alpha=0.7)
    ax2.bar(x + width/2, talib_means, width, label='TA-Lib', color='lightcoral', alpha=0.7)
    
    ax2.set_title('平均执行时间比较')
    ax2.set_ylabel('时间 (秒)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(func_labels, rotation=45)
    ax2.legend()
    
    # 3. 性能比率 (TA-Lib / TA-Box)
    ax3 = axes[1, 0]
    ratios = []
    ratio_labels = []
    
    for func in functions:
        tabox_data = df[(df['function'] == func) & (df['type'] == 'tabox')]
        talib_data = df[(df['function'] == func) & (df['type'] == 'talib')]
        
        if not tabox_data.empty and not talib_data.empty:
            ratio = talib_data['mean_time'].iloc[0] / tabox_data['mean_time'].iloc[0]
            ratios.append(ratio)
            ratio_labels.append(func)
    
    colors = ['green' if r > 1 else 'red' for r in ratios]
    ax3.bar(ratio_labels, ratios, color=colors, alpha=0.7)
    ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax3.set_title('性能比率 (TA-Lib / TA-Box)')
    ax3.set_ylabel('比率')
    ax3.set_xticklabels(ratio_labels, rotation=45)
    ax3.text(0.02, 0.98, '>1: TA-Box更快\n<1: TA-Lib更快', 
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. 统计摘要
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 计算统计信息
    tabox_faster = sum(1 for r in ratios if r > 1)
    talib_faster = sum(1 for r in ratios if r < 1)
    equal = sum(1 for r in ratios if abs(r - 1) < 0.01)
    
    avg_ratio = np.mean(ratios)
    max_ratio = np.max(ratios)
    min_ratio = np.min(ratios)
    
    summary_text = f"""
统计摘要:
总函数数量: {len(ratios)}
TA-Box更快: {tabox_faster} ({tabox_faster/len(ratios)*100:.1f}%)
TA-Lib更快: {talib_faster} ({talib_faster/len(ratios)*100:.1f}%)
性能相当: {equal} ({equal/len(ratios)*100:.1f}%)

平均性能比率: {avg_ratio:.3f}
最大性能比率: {max_ratio:.3f}
最小性能比率: {min_ratio:.3f}
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印详细结果
    print("\n详细性能比较:")
    print("=" * 80)
    print(f"{'函数名':<15} {'TA-Box平均时间':<15} {'TA-Lib平均时间':<15} {'比率':<10} {'更快':<10}")
    print("=" * 80)
    
    for func in functions:
        tabox_data = df[(df['function'] == func) & (df['type'] == 'tabox')]
        talib_data = df[(df['function'] == func) & (df['type'] == 'talib')]
        
        if not tabox_data.empty and not talib_data.empty:
            tabox_time = tabox_data['mean_time'].iloc[0]
            talib_time = talib_data['mean_time'].iloc[0]
            ratio = talib_time / tabox_time
            faster = "TA-Box" if ratio > 1 else "TA-Lib"
            
            print(f"{func:<15} {tabox_time:<15.6f} {talib_time:<15.6f} {ratio:<10.3f} {faster:<10}")

def main():
    """
    主函数 - 从文件读取benchmark结果
    """
    print("从文件读取benchmark结果...")
    
    # 检查是否有benchmark输出文件
    output_file = 'benchmark_output.txt'
    
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            output = f.read()
    else:
        print(f"未找到 {output_file} 文件")
        print("请先运行 benchmark_all.py 并将输出重定向到文件:")
        print("python benchmark_all.py > benchmark_output.txt")
        return
    
    print("解析benchmark结果...")
    
    # 解析结果
    results = parse_benchmark_output(output)
    
    if results:
        print(f"找到 {len(results)} 个benchmark结果")
        
        # 创建比较图表
        create_comparison_charts(results)
        
        # 保存原始数据
        df = pd.DataFrame(results)
        df.to_csv('benchmark_results.csv', index=False)
        print("\n结果已保存到 benchmark_results.csv")
        print("图表已保存到 benchmark_comparison.png")
    else:
        print("未找到有效的benchmark结果")

if __name__ == '__main__':
    main() 