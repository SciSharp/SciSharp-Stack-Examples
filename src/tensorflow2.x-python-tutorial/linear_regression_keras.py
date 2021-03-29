import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

X = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                         7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
Y = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                         2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

tf.config.run_functions_eagerly(True)

input = keras.layers.Input(shape=(1), name="data_in")

output = Dense(1)(input)
model = keras.Model(inputs=input, outputs=output)

# 选定loss函数和优化器
model.compile(loss='mse', optimizer=keras.optimizers.SGD(0.005), metrics='acc')

model.summary()

model.fit(X, Y, epochs=100)

W, b = model.layers[1].get_weights()

print("weight: ", W, "bias", b)