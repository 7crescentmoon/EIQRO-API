import os
import tensorflow as tf

# Define the directory where the model is stored
model_dir = "model"

# Define the filename of the model
model_filename = "model_mm9_v2.h5"

# Construct the full path to the model file by joining the directory and filename
model_path = os.path.join(model_dir, model_filename)

# Load the TensorFlow Keras model from the specified file path
model = tf.keras.models.load_model(model_path)

class_names = [ 'ain', 'alif', 'ba', 'dal', 'dhod', 'dzal',
                'dzho', 'fa', 'ghoin', 'ha', 'ha\'', 'hamzah', 'jim',
                'kaf', 'kho', 'lam', 'lamalif', 'mim', 'nun', 'qof',
                'ro', 'shod', 'sin', 'syin', 'ta', 'tho', 'tsa',
                'wawu', 'ya', 'zain']
