import numpy as np
import tensorflow as tf
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

warnings.simplefilter(action='ignore', category=FutureWarning)

# test for and update gpu
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# create out train, test and validation data
#data source https://www.kaggle.com/c/dogs-vs-cats/

#use these parameters to define the size of the train, validation and tes sets (Note: max is 2500)
train_size = 1000
valid_size = 200
test_size = 100

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
    .flow_from_directory(directory=test_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10, shuffle=False)

assert train_batches.n == train_size * 2
assert valid_batches.n == valid_size * 2
assert test_batches.n == test_size * 2
assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 2


"""
imgs, labels = train_batches.next()
dlplots.plotImages(imgs)
print(labels)
"""
# download base vgg16 model
vgg16_model = tf.keras.applications.vgg16.VGG16()
vgg16_model.summary()

print(commonlib.count_params(vgg16_model))

# create a sequential model and populate it will all but the last layer from the vgg16 model
model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

model.summary()
print(commonlib.count_params(vgg16_model))
for layer in model.layers:
    layer.trainable = False

#add out 2 output classification layer
model.add(Dense(units=2, activation='softmax'))
model.summary()

#train test and create output
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches, validation_data=valid_batches, epochs=4, verbose=2)

test_images, test_labels = next(test_batches)
dlplots.plotImages(test_images)


# test_batches.classes
predictions = model.predict(x=test_batches, verbose=0)
#predictions = np.round(predictions)
cm = confusion_matrix(y_true=test_batches.labels, y_pred=np.argmax(predictions, axis=-1))
print("classes")
print(test_batches.labels)
print("predictions")
print(np.argmax(predictions, axis=-1))
cm_plot_labels = ['cat', 'dog']
dlplots.plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='confusion in a jar')
