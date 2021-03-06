import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import random
import dlplots


gen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    zoom_range=0.1,
    channel_shift_range=10.0,
    horizontal_flip=True)

os.chdir('data_augmentation/')

chosen_image = random.choice(os.listdir('../dogs-vs-cats/data/dogs-vs-cats/train/dog'))
image_path = '../dogs-vs-cats/data/dogs-vs-cats/train/dog/' + chosen_image

assert os.path.isfile(image_path)

image = np.expand_dims(plt.imread(image_path), 0)
plt.imshow(image[0])
plt.savefig("augmentation.png")
plt.close()

# aug_iter = gen.flow(image)
aug_iter = gen.flow(image, save_to_dir='./augmented_images')
aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(10)]
dlplots.plotImages(aug_images, 'augmented_images')

