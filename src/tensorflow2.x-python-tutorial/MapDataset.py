from __future__ import absolute_import, division, print_function

# Import TensorFlow v2.
import tensorflow as tf

tf.autograph.set_verbosity(5, alsologtostdout=True)

def add(x):
    print("debug test test")
    return x * 2

dataset = tf.data.Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]
dataset = dataset.map(add)
for item in dataset:
    print(item)