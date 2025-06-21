# %%
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

import os

os.chdir("ops")
os.system("ld -r -b binary -o meta.o meta.json")
os.system("ld -r -b binary -o kernel_map.o kernel_map.json")
os.chdir("..")

setup(
    name='ops',
    ext_modules=[
        CUDAExtension(
            name='ops',
            sources=glob.glob('ops/*.cu') + glob.glob('ops/*.cpp') + glob.glob('ops/*.cc'), 
            extra_compile_args={
                'nvcc': ['-lineinfo'],
            },
            extra_link_args=['-lcuda', 'ops/meta.o', 'ops/kernel_map.o'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
