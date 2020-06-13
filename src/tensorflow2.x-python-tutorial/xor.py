import tensorflow as tf
import numpy as np
import math

# input X vector
num_hidden = 8
learning_rate = 0.01
features = tf.constant([[1, 0], [1, 1], [0, 0], [0, 1]], dtype=tf.float32)
labels = tf.constant([[0], [1], [1], [0]], dtype=tf.float32)

hidden_weights = tf.Variable(tf.random.truncated_normal([2, num_hidden], stddev= math.sqrt(2)))
optimizer = tf.optimizers.SGD(learning_rate);

training_steps = 10000
display_step = 1000
for step in range(1, training_steps + 1):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        hidden_activations = tf.nn.relu(tf.matmul(features, hidden_weights));
        output_weights = tf.Variable(tf.random.truncated_normal(
                        [num_hidden, 1],
                        seed= 17,
                        stddev= 1/math.sqrt(num_hidden)))
        logits = tf.matmul(hidden_activations, output_weights);

        predictions = tf.tanh(tf.squeeze(logits));
        loss = tf.reduce_mean(tf.square(predictions - tf.cast(labels, tf.float32)), name="loss")

    # Compute gradients.
    gradients = g.gradient(loss, [output_weights])
    
    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, [output_weights]))

    if step % display_step == 0:
        print("step: %i, loss: %f" % (step, loss))

print ("Complete")