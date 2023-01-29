/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using System;
using System.Linq;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Examples;

/// <summary>
/// Simple vanilla neural net solving the famous XOR problem
/// https://github.com/amygdala/tensorflow-workshop/blob/master/workshop_sections/getting_started/xor/README.md
/// </summary>
public class NeuralNetXorKeras : SciSharpExample, IExample
{
    public ExampleConfig InitConfig()
        => Config = new ExampleConfig
        {
            Name = "NN XOR in Keras",
            Enabled = true
        };

    public bool Run()
    {
        tf.enable_eager_execution();

        var x = np.array(new float[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
        var y = np.array(new float[,] { { 0 }, { 1 }, { 1 }, { 0 } });

        var model = keras.Sequential();
        model.add(keras.Input(2));
        model.add(keras.layers.Dense(32, keras.activations.Relu));
        model.add(keras.layers.Dense(1, keras.activations.Sigmoid));
        model.compile(optimizer: keras.optimizers.Adam(),
            loss: keras.losses.MeanSquaredError(),
            new[] { "accuracy" });
        model.fit(x, y, 1, 100);
        model.evaluate(x, y);
        Tensor result = model.predict(x, 4);
        return result.ToArray<float>() is [< 0.5f, > 0.5f, > 0.5f, < 0.5f];
    }
}
