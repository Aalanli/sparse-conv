# sparse-conv

Sparse-conv is a minimal 3D sparse convolution library that leverages triton kernel generation to match or exceed the performance of publicly available libraries like spconv or torchsparse.

Sparse convolution proceeds by 2 steps:
1. Generate indices
- This is implemented via a GPU hashmap in CUDA
2. Implicit Gemm
- This is a fused matrix multiplication implemented in Triton.

Sparse-conv uses triton to ahead-of-time compile implicit gemm kernels, embeds the PTX directly into the binary, and thus is suitable for onboard inference where JIT is unacceptable.


## Building the library:
```
pip install -r requirements.txt
python setup.py build_ext
python generate_ptx.py --sm 86 89 --dtype fp16 fp32
```


## Running the benchmark

building torchsparse
```
sudo apt-get install libsparsehash-dev
git clone https://github.com/mit-han-lab/torchsparse
cd torchsparse
pip install -r requirements.txt
python setup.py install
```

Note that spconv-cu120 requires numpy==1.26.4, newer versions may cause `floating point exception`
```
pip install numpy==1.26.4
pip install spconv-cu120==2.3.6
```


```
./run_bench.sh <prefix>
```
where prefix can be sm86_a6000
