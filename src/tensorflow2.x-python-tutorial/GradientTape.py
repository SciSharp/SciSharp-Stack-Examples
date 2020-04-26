
import tensorflow as tf

w = tf.constant(1.5)
with tf.GradientTape() as tape:
    tape.watch(w)
    loss = w * w

grad = tape.gradient(loss, w)

x = tf.Variable(1)
y = tf.Variable(2)

with tf.GradientTape() as t:
  t.watch(x)
  z = tf.subtract(2*x, y)
dz_dx = t.gradient(z, [x, y])

# Derivative of z with respect to the original input tensor x
dz_dx = t.gradient(z, x)
print(dz_dx[0][0].numpy())

for i in [0, 1]:
  for j in [0, 1]:
    assert dz_dx[i][j].numpy() == 8.0
    print(dz_dx[i][j].numpy())