namespace TensorFlowNET.Examples.FSharp

open NumSharp
open type Tensorflow.Binding

/// A very simple "hello world" using TensorFlow v2 tensors.
/// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/1_Introduction/helloworld.ipynb
module HelloWorld =
    let private run () =
        // Eager model is enabled by default.
        tf.enable_eager_execution()

        (* Create a Constant op
           The op is added as a node to the default graph.

           The value returned by the constructor represents the output
           of the Constant op. *)
        let str = "Hello, TensorFlow.NET!"
        let hello = tf.constant(str)

        // tf.Tensor: shape=(), dtype=string, numpy=b'Hello, TensorFlow.NET!'
        print(hello);

        let tensor = NDArray.AsStringArray(hello.numpy()).[0]

        str = tensor

    let Example =
        { Config = ExampleConfig.Create("Hello World", priority0 = 1)
          Run = run }
