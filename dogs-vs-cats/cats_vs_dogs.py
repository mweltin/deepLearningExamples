import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import os
import shutil
import random
import glob
import warnings
import dlplots

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

train_size = 2000
valid_size = 100
test_size = 100
    for c in random.sample(glob.glob('../../train/cat*'), train_size):
    shutil.copy(c, 'train/cat')
for c in random.sample(glob.glob('../../train/dog*'), train_size):
    shutil.copy(c, 'train/dog')
for c in random.sample(glob.glob('../../train/cat*'), valid_size):
    shutil.copy(c, 'valid/cat')
for c in random.sample(glob.glob('../../train/dog*'), valid_size):
    shutil.copy(c, 'valid/dog')
for c in random.sample(glob.glob('../../train/cat*'), test_size):
    shutil.copy(c, 'test/cat')
for c in random.sample(glob.glob('../../train/dog*'), test_size):
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

assert train_batches.n == train_size * 2
assert valid_batches.n == valid_size * 2
assert test_batches.n == test_size * 2
assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 2


"""
imgs, labels = train_batches.next()
dlplots.plotImages(imgs)
print(labels)
"""

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=2, activation='softmax')
])
model.summary()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches, validation_data=valid_batches, epochs=2, verbose=2)

test_images, test_labels = test_batches.next()
dlplots.plotImages(test_images)
print(test_labels)

test_batches.classes
predictions = model.predict(x=test_batches, verbose=0)
np.round(predictions)
cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=1))
test_batches.class_indices
cm_plot_labels = ['cats', 'dogs']
dlplots.plot_confusion_matrix(cm, classes=cm_plot_labels, title='confusion in a jar')
