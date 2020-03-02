
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/1_Introduction/helloworld.ipynb

import tensorflow as tf
# Create a Tensor.
hello = tf.constant("hello world")
print(hello)

# To access a Tensor value, call numpy().
val = hello.numpy()

print(val)