import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
import setproctitle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

setproctitle.setproctitle("[k4ke] tf_test")

# print(tf.test.is_gpu_available())

model = keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
model.add(layers.Embedding(input_dim = 1000, output_dim = 64))

# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(128))

# Add a Dense layer with 10 units
model.add(layers.Dense(10))

model.summary()



