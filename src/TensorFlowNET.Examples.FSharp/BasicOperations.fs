namespace TensorFlowNET.Examples.FSharp

open type Tensorflow.Binding

module BasicOperations =

    let private run() =
        tf.enable_eager_execution()

        // Define tensor constants.
        let a = tf.constant(2)
        let b = tf.constant(3)
        let c = tf.constant(5)

        // Various tensor operations.
        // Note: Tensors also support operators (+, *, ...)
        let add = tf.add(a, b)
        let sub = tf.subtract(a, b)
        let mul = tf.multiply(a, b)
        let div = tf.divide(a, b)

        // Access tensors value.
        print("add =", add.numpy())
        print("sub =", sub.numpy())
        print("mul =", mul.numpy())
        print("div =", div.numpy())

        // Some more operations.
        let mean = tf.reduce_mean([| a; b; c |])
        let sum = tf.reduce_sum([| a; b; c |])

        // Access tensors value.
        print("mean =", mean.numpy())
        print("sum =", sum.numpy())

        // Matrix multiplications.
        let matrix1 = tf.constant(array2D [ [ 1; 2 ]; [ 3; 4 ] ])
        let matrix2 = tf.constant(array2D [ [ 5; 6 ]; [ 7; 8 ] ])
        let product = tf.matmul(matrix1, matrix2)
        // Convert Tensor to Numpy.
        print("product =", product.numpy())

        true

    let Example =
        { Config = ExampleConfig.Create ("Basic Operations", priority0 = 2)
          Run = run
        }

