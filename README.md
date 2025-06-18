# sparse-conv

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
