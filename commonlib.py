import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


def count_params(model):
    non_trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.non_trainable_weights])
    trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_weights])
    return {'non_trainable_params': non_trainable_params, 'trainable_params': trainable_params}


def prepare_image(file):
    img_path = 'dogs-vs-cats/data/MobileNet-samples/'
    img = image.load_img(img_path + file, target_size=(244, 244))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
