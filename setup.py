from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

setup(
    name='ops',
    ext_modules=[
        CUDAExtension(
            name='ops',
            sources=glob.glob('ops/*.cu') + glob.glob('ops/*.cpp') + glob.glob('ops/*.cc'), 
            extra_compile_args={
                'nvcc': ['-lineinfo'],
            },
            extra_link_args=['-lcuda']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
