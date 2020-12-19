from __future__ import absolute_import, division, print_function

# Import TensorFlow v2.
import tensorflow as tf
from tensorflow.python.data.experimental.ops import cardinality

tf.autograph.set_verbosity(10, alsologtostdout=True)
tf.config.run_functions_eagerly(True)

def add(x):
    print("debug test test")
    return x + 1

dataset = tf.data.Dataset.range(10)  # ==> [ 1, 2, 3, 4, 5 ]
dataset = dataset.map(add)
card = cardinality.cardinality(dataset)
for item in dataset:
    print(item)
