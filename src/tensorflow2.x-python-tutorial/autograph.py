import tensorflow as tf

tf.autograph.set_verbosity(10, alsologtostdout=True)

# Let's implement a simple Linear layer (mx+b) using tf.function 
@tf.function
def add(a,b):
  return 10

result = add(1, 2)
print(result)
