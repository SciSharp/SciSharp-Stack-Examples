import numpy as np
import tensorflow as tf
from tensorflow import keras

model = tf.keras.Sequential()
layer = tf.keras.layers.Embedding(7, 2, input_length=4)
model.add(layer)
# The model will take as input an integer matrix of size (batch,
# input_length), and the largest integer (i.e. word index) in the input
# should be no larger than 999 (vocabulary size).
# Now model.output_shape is (None, 10, 64), where `None` is the batch
# dimension.
model.compile('rmsprop', 'mse')

input_array = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])
# np.random.randint(4, size=(3, 4))
output_array = model.predict(input_array)
print(output_array.shape)