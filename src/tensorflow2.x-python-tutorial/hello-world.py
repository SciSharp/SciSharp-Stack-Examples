
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/1_Introduction/helloworld.ipynb


import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
a = tf.constant(-1)
# Create a `Sequential` model and add a Dense layer as the first layer.
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(16,)))
model.add(tf.keras.layers.Dense(32, activation='relu'))
# Now the model will take as input arrays of shape (None, 16)
# and output arrays of shape (None, 32).
# Note that after the first layer, you don't need to specify
# the size of the input anymore:
model.add(tf.keras.layers.Dense(32))
print(model.output_shape)

model = tf.keras.Sequential()
dense_layer = tf.keras.layers.Dense(5, input_shape=(3,))
eager = tf.executing_eagerly()
model.add(dense_layer)
# model.add(tf.keras.layers.Softmax())
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