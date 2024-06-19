from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="reduce_cuda",
    ext_modules=[
        CUDAExtension(
            name="reduce_cuda",
            sources=['src\reduce.cu'],
            extra_compile_args={
                'cxx': ['-03'],
                'nvcc':[
                    '-03',
                    '--gencode', 'arch=compute_80, code=sm_80',
                    '--gencode', 'arch=compute_86, code=sm_86'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)