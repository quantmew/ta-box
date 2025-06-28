import glob
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import sys
import os
import io
from contextlib import redirect_stdout

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
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

def run_all_benchmarks():
    """
    运行所有benchmark并收集输出
    """
    bench_files = glob.glob("benchmark/bench_*.py")
    all_output = []
    
    for i, file in enumerate(bench_files):
        print(f"Running {file} [{i+1}/{len(bench_files)}]")
        
        # 运行benchmark并捕获输出
        output = io.StringIO()
        
        # 保存当前工作目录
        original_cwd = os.getcwd()
        original_sys_path = sys.path.copy()
        
        try:
            # 切换到benchmark目录
            os.chdir(os.path.dirname(file))
            
            # 添加项目根目录到sys.path
            project_root = os.path.abspath(os.path.join(os.path.dirname(file), '..'))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            with redirect_stdout(output):
                # 读取并执行benchmark文件
                with open(os.path.basename(file), 'r', encoding='utf-8') as f:
                    code = f.read()
                
                # 创建一个新的命名空间来执行代码
                namespace = {
                    '__file__': os.path.abspath(file),
                    '__name__': '__main__',
                    '__builtins__': __builtins__
                }
                
                exec(code, namespace)
                
        except Exception as e:
            print(f"运行 {file} 时出错: {e}")
            continue
        finally:
            # 恢复原始状态
            os.chdir(original_cwd)
            sys.path = original_sys_path
            
        result = output.getvalue()
        
        if result:
            all_output.append(result)
            print(result.strip())
    
    return '\n'.join(all_output)

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
    
    # 准备数据
    tabox_means = []
    talib_means = []
    tabox_totals = []
    talib_totals = []
    func_labels = []
    ratios = []
    
    for func in functions:
        tabox_data = df[(df['function'] == func) & (df['type'] == 'tabox')]
        talib_data = df[(df['function'] == func) & (df['type'] == 'talib')]
        
        if not tabox_data.empty and not talib_data.empty:
            tabox_means.append(float(tabox_data['mean_time'].tolist()[0]))
            talib_means.append(float(talib_data['mean_time'].tolist()[0]))
            tabox_totals.append(float(tabox_data['total_time'].tolist()[0]))
            talib_totals.append(float(talib_data['total_time'].tolist()[0]))
            func_labels.append(func)
            
            ratio = float(talib_data['mean_time'].tolist()[0]) / float(tabox_data['mean_time'].tolist()[0])
            ratios.append(ratio)
    
    # 1. 总时间比较图
    plt.figure(figsize=(16, 8))
    x = np.arange(len(func_labels))
    width = 0.35
    
    plt.bar(x - width/2, tabox_totals, width, label='TA-Box', color='skyblue', alpha=0.7)
    plt.bar(x + width/2, talib_totals, width, label='TA-Lib', color='lightcoral', alpha=0.7)
    
    plt.title('总执行时间比较', fontsize=16, fontweight='bold')
    plt.ylabel('时间 (秒)', fontsize=12)
    plt.xlabel('函数名', fontsize=12)
    plt.xticks(x, func_labels, rotation=90, fontsize=10)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('benchmark_total_time.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 平均时间比较图
    plt.figure(figsize=(16, 8))
    
    plt.bar(x - width/2, tabox_means, width, label='TA-Box', color='skyblue', alpha=0.7)
    plt.bar(x + width/2, talib_means, width, label='TA-Lib', color='lightcoral', alpha=0.7)
    
    plt.title('平均执行时间比较', fontsize=16, fontweight='bold')
    plt.ylabel('时间 (秒)', fontsize=12)
    plt.xlabel('函数名', fontsize=12)
    plt.xticks(x, func_labels, rotation=90, fontsize=10)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('benchmark_mean_time.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. 性能比率图
    plt.figure(figsize=(16, 8))
    
    colors = ['green' if r > 1 else 'red' for r in ratios]
    bars = plt.bar(func_labels, ratios, color=colors, alpha=0.7)
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=2)
    
    # 在柱状图上添加数值标签
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{ratio:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.title('性能比率 (TA-Lib / TA-Box)', fontsize=16, fontweight='bold')
    plt.ylabel('比率', fontsize=12)
    plt.xlabel('函数名', fontsize=12)
    plt.xticks(rotation=90, fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 添加图例说明
    plt.text(0.02, 0.98, '>1: TA-Box更快\n<1: TA-Lib更快', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=12)
    
    plt.tight_layout()
    plt.savefig('benchmark_ratio.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. 统计摘要图
    plt.figure(figsize=(10, 6))
    plt.axis('off')
    
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
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
             fontsize=14, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.title('性能比较统计摘要', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('benchmark_summary.png', dpi=300, bbox_inches='tight')
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
            tabox_time = float(tabox_data['mean_time'].tolist()[0])
            talib_time = float(talib_data['mean_time'].tolist()[0])
            ratio = talib_time / tabox_time
            faster = "TA-Box" if ratio > 1 else "TA-Lib"
            
            print(f"{func:<15} {tabox_time:<15.6f} {talib_time:<15.6f} {ratio:<10.3f} {faster:<10}")
    
    print(f"\n图表已保存为:")
    print("- benchmark_total_time.png (总执行时间比较)")
    print("- benchmark_mean_time.png (平均执行时间比较)")
    print("- benchmark_ratio.png (性能比率)")
    print("- benchmark_summary.png (统计摘要)")

def main():
    """
    主函数
    """
    print("开始运行所有benchmark...")
    
    # 运行所有benchmark
    output = run_all_benchmarks()
    
    print("\n解析benchmark结果...")
    
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
    else:
        print("未找到有效的benchmark结果")

if __name__ == '__main__':
    main() 