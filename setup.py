#!/usr/bin/env python
import os
from setuptools import setup, find_packages

setup(
    name="rotation_matrix_cuda",
    version="1.0",
    description="rotating multi-channel matrix for 4 directions--0, 90, 180, 270 degrees in order.",
    author="Gao Wei",
    author_email="gaoweichn@126.com",
    # Require cffi.
    install_requires=["cffi>=1.0.0"],
    setup_requires=["cffi>=1.0.0"],
    # Exclude the build files.
    packages=find_packages(exclude=["build"]),
    # Package where to put the extensions. Has to be a prefix of build.py.
    ext_package="",
    # Extensions to compile.
    cffi_modules=[
        os.path.join(os.path.dirname(__file__), "build.py:ffi")
    ],
)