
python bench.py --forward --plot_file="benchmarks/benchmark_${1}_D16.png" --Ds 16
python bench.py --backward --plot_file="benchmarks/benchmark_${1}_D16.png" --Ds 16

python bench.py --forward --plot_file="benchmarks/benchmark_${1}_D32.png" --Ds 32
python bench.py --backward --plot_file="benchmarks/benchmark_${1}_D32.png" --Ds 32

python bench.py --forward --plot_file="benchmarks/benchmark_${1}_D64.png" --Ds 64
python bench.py --backward --plot_file="benchmarks/benchmark_${1}_D64.png" --Ds 64

python bench.py --forward --plot_file="benchmarks/benchmark_${1}_D128.png" --Ds 128
python bench.py --backward --plot_file="benchmarks/benchmark_${1}_D128.png" --Ds 128