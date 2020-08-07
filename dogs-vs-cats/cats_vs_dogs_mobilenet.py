import tensorflow as tf
from tensorflow.keras.applications import imagenet_utils
import warnings
import commonlib
from IPython.display import Image

warnings.simplefilter(action='ignore', category=FutureWarning)

# test for and update gpu
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# download base mobilenet model
mobilenet_model = tf.keras.applications.mobilenet.MobileNet()
mobilenet_model.summary()

Image(filename='data/MobileNet-samples/samson.png', width=300, height=200)
preprocessed_image = commonlib.prepare_image('samson.png')
predictions = mobilenet_model.predict(preprocessed_image)
# helper function to return the 5 most probable results.
results = imagenet_utils.decode_predictions(predictions)
print(results)