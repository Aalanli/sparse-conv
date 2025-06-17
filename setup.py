from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

setup(
    name='ops',
    ext_modules=[
        CUDAExtension(
            name='ops',
            sources=glob.glob('ops/*.cu') + glob.glob('ops/*.cpp'), 
            extra_compile_args={
                'nvcc': ['-lineinfo'],
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)