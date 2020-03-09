using NumSharp;
using System;
using System.Diagnostics;
using Tensorflow;
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
                Name = "Basic Operations",
                Priority = 2
            };

        public bool Run()
        {
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
            print("add =", add.numpy());
            print("sub =", sub.numpy());
            print("mul =", mul.numpy());
            print("div =", div.numpy());

            return true;
        }
    }
}
