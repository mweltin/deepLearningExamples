import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
# %matplotlib inline

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

os.chdir('data/dogs-vs-cats')

if os.path.isdir('./train') is True:
    shutil.rmtree('./train')
if os.path.isdir('./valid') is True:
    shutil.rmtree('./valid')
if os.path.isdir('./test') is True:
    shutil.rmtree('./test')

if os.path.isdir('train/dog') is False:
    os.makedirs('train/dog')
    os.makedirs('train/cat')
    os.makedirs('valid/dog')
    os.makedirs('valid/cat')
    os.makedirs('test/dog')
    os.makedirs('test/cat')

for c in random.sample(glob.glob('../../train/cat*'), 500):
    shutil.copy(c, 'train/cat')
for c in random.sample(glob.glob('../../train/dog*'), 500):
    shutil.copy(c, 'train/dog')
for c in random.sample(glob.glob('../../train/cat*'), 100):
    shutil.copy(c, 'valid/cat')
for c in random.sample(glob.glob('../../train/dog*'), 100):
    shutil.copy(c, 'valid/dog')
for c in random.sample(glob.glob('../../train/cat*'), 50):
    shutil.copy(c, 'test/cat')
for c in random.sample(glob.glob('../../train/dog*'), 50):
    shutil.copy(c, 'test/dog')

os.chdir('../../')

train_path = 'data/dogs-vs-cats/train'
valid_path = 'data/dogs-vs-cats/valid'
test_path = 'data/dogs-vs-cats/test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10)

assert train_batches.n == 1000
assert valid_batches.n == 200
assert test_batches.n == 100
assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 2
imgs, labels = train_batches.next()
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    # plt.show()
    plt.savefig("dogs-v-cat.png")


plotImages(imgs)
print(labels)
