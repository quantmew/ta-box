import glob
bench_files = glob.glob("benchmark/bench_*.py")

for i, file in enumerate(bench_files):
    print(f"Running {file} [{i+1}/{len(bench_files)}]")
    exec(open(file).read())