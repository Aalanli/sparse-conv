# sparse-conv

building torchsparse
```
sudo apt-get install libsparsehash-dev
git clone https://github.com/mit-han-lab/torchsparse
cd torchsparse
pip install -r requirements.txt
python setup.py install

```

Note that spconv-cu120 requires numpy==1.26.4, newer versions may cause `floating point exception`


# triton AOT

```
python -m triton.tools.compile test_jit.py -n=saxpy_kernel -w=2 -ns=1 -s="*fp32:16, *fp32:16, fp32, *fp32:16, i32, 128" -on=saxpy -g 1,1,1
```
