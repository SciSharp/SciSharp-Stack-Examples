
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/1_Introduction/helloworld.ipynb

import tensorflow as tf
import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, input_shape=(3,)),
    tf.keras.layers.Softmax()])
model.save('/tmp/model')
loaded_model = tf.keras.models.load_model('/tmp/model')
x = tf.random.uniform((10, 3))
assert np.allclose(model.predict(x), loaded_model.predict(x))

tensor = [0, 1, 2, 3]
mask = np.array([True, False, True, False])
masked = tf.boolean_mask(tensor, mask)

a = tf.constant(0.0);
b = 2.0 * a;

X = tf.placeholder(tf.double)
W = tf.constant(1.0)
mul = tf.multiply(X, W)

ones = tf.zeros([300, 400], tf.int32) 

x = tf.Variable(10, name = "x");
for i in range(0, 5):
    x = x + 1;

# Create a Tensor.
hello = tf.constant("hello world")
print(hello)

# To access a Tensor value, call numpy().
val = hello.numpy()

print(val)