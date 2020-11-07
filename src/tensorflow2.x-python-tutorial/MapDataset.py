from __future__ import absolute_import, division, print_function

# Import TensorFlow v2.
import tensorflow as tf

tf.autograph.set_verbosity(10, alsologtostdout=True)

def add(x):
    print("debug test test")
    return x + 1

dataset = tf.data.Dataset.range(1, 3)  # ==> [ 1, 2, 3, 4, 5 ]
dataset = dataset.map(add)
for item in dataset:
    print(item)