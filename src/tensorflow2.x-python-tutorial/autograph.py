import tensorflow as tf

tf.compat.v1.disable_eager_execution()

tf.autograph.set_verbosity(10, alsologtostdout=True)

# Let's implement a simple Linear layer (mx+b) using tf.function 
# @tf.function
def min(a,b):
    if(a > b):
        return a
    else:
        return b

tf.autograph.to_graph(min)
input_a = tf.constant(1)
input_b = tf.constant(2)
result = min(input_a, input_b)
print(result)


i = tf.constant(1)
def c(x, y): 
    return tf.less(x + y, 10)
def b(x, y):
   return [tf.add(x, 1), tf.add(y, 1)]
r = tf.while_loop(c, b, [2, 3])