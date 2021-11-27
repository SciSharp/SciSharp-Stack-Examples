using static Tensorflow.Binding;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Basic tensor operations using TensorFlow v2.
    /// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/1_Introduction/basic_operations.ipynb
    /// </summary>
    public class BasicOperations : SciSharpExample, IExample
    {
        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "Basic Operations"
            };

        public bool Run()
        {
            tf.enable_eager_execution();

            // Define tensor constants.
            var a = tf.constant(2);
            var b = tf.constant(3);
            var c = tf.constant(5);

            // Various tensor operations.
            // Note: Tensors also support operators (+, *, ...)
            var add = tf.add(a, b);
            var sub = tf.subtract(a, b);
            var mul = tf.multiply(a, b);
            var div = tf.divide(a, b);

            // Access tensors value.
            print($"{(int)a} + {(int)b} = {(int)add}");
            print($"{(int)a} - {(int)b} = {(int)sub}");
            print($"{(int)a} * {(int)b} = {(int)mul}");
            print($"{(int)a} / {(int)b} = {(double)div}");

            // Some more operations.
            var mean = tf.reduce_mean(tf.constant(new[] { a, b, c }));
            var sum = tf.reduce_sum(tf.constant(new[] { a, b, c }));

            // Access tensors value.

            print("mean =", mean.numpy());
            print("sum =", sum.numpy());

            // Matrix multiplications.
            var matrix1 = tf.constant(new float[,] { { 1, 2 }, { 3, 4 } });
            var matrix2 = tf.constant(new float[,] { { 5, 6 }, { 7, 8 } });
            var product = tf.matmul(matrix1, matrix2);
            // Convert Tensor to Numpy.
            print("product =", product.numpy());

            return true;
        }
    }
}
