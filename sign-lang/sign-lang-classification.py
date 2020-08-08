import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import os
import shutil
import random
import glob
import warnings
import dlplots
import commonlib

os.chdir('sign-lang/data')

if os.path.isdir('./train') is True:
    shutil.rmtree('./train')
if os.path.isdir('./valid') is True:
    shutil.rmtree('./valid')
if os.path.isdir('./test') is True:
    shutil.rmtree('./test')

if os.path.isdir('train/0/') is False:
    os.mkdir('train')
    os.mkdir('valid')
    os.mkdir('test')

for i in range(0, 10):
    shutil.copytree(f'org_data/{i}', f'train/{i}')
    os.mkdir(f'valid/{i}')
    os.mkdir(f'test/{i}')

for i in range(0, 10):
    valid_sample = random.sample(os.listdir(f'train/{i}'), 30)
    for j in valid_sample:
        shutil.move(f'train/{i}/{j}', f'valid/{i}')

    test_samples = random.sample(os.listdir(f'train/{i}'), 5)
    for k in test_samples:
        shutil.move(f'train/{i}/{k}', f'test/{i}')

os.chdir('../')

train_path = 'data/train'
test_path = 'data/test'
valid_path = 'data/valid'

train_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=train_path,
                                                                                                 target_size=(224, 224),
                                                                                                 batch_size=10)
valid_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=valid_path,
                                                                                                 target_size=(224, 224),
                                                                                                 batch_size=10)
test_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=test_path,
                                                                                                 target_size=(224, 224),
                                                                                                 batch_size=10,
                                                                                                 shuffle=False)

assert train_batches.n == 1712
assert valid_batches.n == 300
assert test_batches.n == 50
assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 10

mobile = tf.keras.applications.mobilenet.MobileNet()
mobile.summary()
params = commonlib.count_params(mobile)
assert params['non_trainable_params'] == 21888
assert params['trainable_params'] == 4231976

x = mobile.layers[-6].output
output = Dense(units=10, activation='softmax')(x)

model = Model(inputs=mobile.input, outputs=output)

for layer in model.layers[:-23]:
    layer.trainable = False

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches, validation_data=valid_batches, epochs=10, verbose=2)

# true labels
test_labels = test_batches.classes
# get predictions
predictions = model.predict(x=test_batches, verbose=0)

cm = confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))
print(test_batches.class_indices)

cm_plot_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# generate confusion matrix png
dlplots.plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='new confusion matrix')
