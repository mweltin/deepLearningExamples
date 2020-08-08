Base install:
GPU: GeForce RTX 2080 Ti
Driver Version: 440.100      
CUDA Version: 10.2
CUDNN: 10.1 v7.6.4.38
Tensorflow: 2.3.0
Python: 3.8.2

PyCharm Setup:
Add the LD_PATH to your run configuration python template's environment variables
PYTHONUNBUFFERED=1;LD_LIBRARY_PATH=/usr/lib/cuda/include:/usr/lib/cuda/lib64
For Run and Debug configurations set the directory with commonlib.py as your working directory.
