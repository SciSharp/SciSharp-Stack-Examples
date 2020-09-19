import tensorflow as tf

tf.autograph.set_verbosity(10, alsologtostdout=True)

# Let's implement a simple Linear layer (mx+b) using tf.function 
# @tf.function
def add(a,b):
  return a + b

input_a = tf.constant(1)
input_b = tf.constant(2)
result = add(input_a, input_b)
print(result)
