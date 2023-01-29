using System;
using System.Linq;
using static Tensorflow.Binding;

namespace TensorFlowNET.Examples;

/// <summary>
/// A very simple "hello world" using TensorFlow v2 tensors.
/// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/1_Introduction/helloworld.ipynb
/// </summary>
public class HelloWorld : SciSharpExample, IExample
{
    public ExampleConfig InitConfig()
        => Config = new ExampleConfig
        {
            Name = "Hello World"
        };

    public HelloWorld()
    {
        str = string.Join("", Enumerable.Range(0, 1024 * 1024 * 20).Select(x => "X"));
    }
    public static string str;
    public bool Run()
    {
        // Eager model is enabled by default.
        tf.enable_eager_execution();

        /* Create a Constant op
           The op is added as a node to the default graph.
        
           The value returned by the constructor represents the output
           of the Constant op. */
        // var str = string.Join("", Enumerable.Range(0, 1024 * 1024 * 20).Select(x => "X"));
        using var hello = tf.constant(str);
        
        // tf.Tensor: shape=(), dtype=string, numpy=b'Hello, TensorFlow.NET!'
        // print(hello);

        var tensor = hello.StringData();

        //return tensor.ToString() == $"'{str}'";
        return true;
    }
}
