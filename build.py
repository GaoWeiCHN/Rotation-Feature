import os
import torch
from torch.utils.ffi import create_extension
import sys

this_file = os.path.dirname(__file__)

sources = []
headers = []
extra_objects = []
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['rotation_feature/src/rotationFeature.c']
    headers += ['rotation_feature/src/rotationFeature.h']
    extra_objects += ['rotation_feature/src/rotationFeatureKernel.cu.o']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

extra_compile_args = None if sys.platform == 'darwin' else ['-fopenmp'] # MacOS does not support 'fopenmp'
ffi = create_extension(
    'rotation_feature._ext.rm',
    package=True,
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    include_dirs=['rotation_feature/src'],
    with_cuda=with_cuda,
    extra_objects=extra_objects,
    extra_compile_args=extra_compile_args,
)

if __name__ == '__main__':
    ffi.build()