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

            // Various tensor operations.
            // Note: Tensors also support operators (+, *, ...)
            var add = tf.add(a, b);
            Debug.Assert(5 == add.numpy());

            return true;
        }
    }
}
