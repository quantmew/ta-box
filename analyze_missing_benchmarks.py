import os
import glob

# 获取所有ta_func函数
ta_func_files = glob.glob("tabox/ta_func/ta_*.py")
ta_func_names = []
for file in ta_func_files:
    if file.endswith('.py') and not file.endswith('_defs.py') and not file.endswith('_utility.py') and not file.endswith('_utils.py'):
        name = os.path.basename(file).replace('.py', '')
        ta_func_names.append(name)

# 获取所有benchmark文件
bench_files = glob.glob("benchmark/bench_*.py")
bench_names = []
for file in bench_files:
    name = os.path.basename(file).replace('.py', '')
    bench_names.append(name)

print("现有的benchmark文件:")
for bench in sorted(bench_names):
    print(f"  {bench}")

print(f"\nta_func函数总数: {len(ta_func_names)}")
print(f"benchmark文件总数: {len(bench_names)}")

# 找出缺失的benchmark
missing_benchmarks = []
for ta_func in ta_func_names:
    # 移除ta_前缀并转换为小写
    func_name = ta_func.replace('ta_', '').lower()
    bench_name = f"bench_{func_name}"
    if bench_name not in bench_names:
        missing_benchmarks.append(ta_func)

print(f"缺失的benchmark数量: {len(missing_benchmarks)}")
print("\n缺失的benchmark函数:")
for func in sorted(missing_benchmarks):
    print(f"  {func}") 