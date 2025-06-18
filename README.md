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

