
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/1_Introduction/basic_operations.ipynb

import tensorflow as tf

# tf.compat.v1.disable_eager_execution()

image = tf.constant([
 [1,0,0,0,0],
 [0,1,0,0,0],
 [0,0,1,0,0],
 [0,0,0,1,0],
 [0,0,0,0,1],
])

image = image[tf.newaxis, ..., tf.newaxis]
image = tf.image.resize(image, [3,5])
image = image[0,...,0]

x = tf.constant([[1.0, -0.5, 3.4], [-2.1, 0, -6.5]])
a = tf.reduce_sum(x)  # 6
a = tf.reduce_sum(x, 0)  # [2, 2, 2]
a = tf.reduce_sum(x, 1)  # [3, 3]
a = tf.reduce_sum(x, 1, keepdims=True)  # [[3], [3]]
a = tf.reduce_sum(x, [0, 1])  # 6

const = tf.constant(['map_and_batch_fusion', 'noop_elimination', 'shuffle_and_repeat_fusion'], 
                             dtype=tf.string, 
                             name="optimizations")

sess = tf.compat.v1.Session()
result = sess.run(const)
print(result)


tensor = tf.linalg.eye(3)

gs = tf.compat.v1.Variable(0, trainable = False, name = "global_step");
gs.assign(10)
# x = tf.Variable(10, name = "x")

# Build a graph
a = tf.constant(4.0)
b = tf.constant(5.0)
c = tf.add(a, b)

with tf.compat.v1.Session() as sess:
    o = sess.run(c)
    print(o)

str = tf.constant('Hello')

a = tf.constant(b'\x41\xff\xd8\xff', dtype=tf.string)

contents = tf.io.read_file('D:\SciSharp\TensorFlow.NET\data\shasta-daisy.jpg')
substr = tf.strings.substr(contents, 0, 3)
b = tf.equal(substr, b'\xff\xd8\xff')

# Define tensor constants.
a = tf.constant(2)
b = tf.constant(3)
c = tf.constant(5)

# Various tensor operations.
# Note: Tensors also support python operators (+, *, ...)
add = tf.add(a, b)
sub = tf.subtract(a, b)
mul = tf.multiply(a, b)
div = tf.divide(a, b)

# Access tensors value.
print("add =", add.numpy())
print("sub =", sub.numpy())
print("mul =", mul.numpy())
print("div =", div.numpy())

# Some more operations.
mean = tf.reduce_mean([a, b, c])
sum = tf.reduce_sum([a, b, c])

# Access tensors value.
print("mean =", mean.numpy())
print("sum =", sum.numpy())

# Matrix multiplications.
matrix1 = tf.constant([[1., 2.], [3., 4.]])
matrix2 = tf.constant([[5., 6.], [7., 8.]])

product = tf.matmul(matrix1, matrix2)

# Convert Tensor to Numpy.
product.numpy()