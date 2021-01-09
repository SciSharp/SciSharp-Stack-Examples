import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.python.platform import gfile
# GRAPH_PB_PATH = 'D:/tmp/TensorflowIssue/TensorflowIssue/model/saved_model.pb'
GRAPH_PB_PATH = 'D:/tmp/TensorFlow.NET/data/saved_model.pb'
with tf.compat.v1.Session() as sess:
   print("load graph")
   with tf.io.gfile.GFile(GRAPH_PB_PATH,'rb') as f:
       graph_def = tf.compat.v1.GraphDef()
       graph_def.ParseFromString(f.read())
   sess.graph.as_default()
   tf.import_graph_def(graph_def, name='')
   graph_nodes=[n for n in graph_def.node]
   names = []
   for t in graph_nodes:
      names.append(t.name)
   print(names)

inputs = keras.Input(shape=(784,))

dense = layers.Dense(64, activation="relu")
x = dense(inputs)

dense = layers.Dense(64, activation="relu")
x = dense(x)

dense = layers.Dense(10)
outputs = dense(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
model.summary();

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)

test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])