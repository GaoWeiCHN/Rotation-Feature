#!/bin/bash
HOME=$(pwd)
echo "Compiling cuda kernels..."
cd $HOME/rotation_feature/src
rm rotationFeatureKernel.cu.o
nvcc -c  -o rotationFeatureKernel.cu.o rotationFeatureKernel.cu -x cu -Xcompiler -fPIC -arch=sm_35
echo "Installing extension..."
cd $HOME
python setup.py clean && python setup.py install