
import tensorflow as tf

x = tf.Variable(3.0, dtype = tf.float32)
with tf.GradientTape() as tape:
    y = tf.square(x)
y_grad = tape.gradient(y, x)

x = tf.Variable(1.0)
tf.split([[1, 2], [3, 4]], num_or_size_splits = 2, axis = 0)
with tf.GradientTape() as t1:
    with tf.GradientTape() as t2:
        y = x * x * x
        print(y)
    dy_dx = t2.gradient(y, x)
    print(dy_dx)
d2y_d2x = t1.gradient(dy_dx, x)
print(d2y_d2x)