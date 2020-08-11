import numpy as np
import tensorflow as tf
import random as rm
import os

# with a keras model you need to set three seeds  to get reproducible results.
# Note that the integers used below can be any int.

os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(45)
rm.seed(234)
tf.random.set_seed(234)

