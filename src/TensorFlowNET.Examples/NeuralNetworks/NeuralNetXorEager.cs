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
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Examples;

/// <summary>
/// Simple vanilla neural net solving the famous XOR problem
/// https://github.com/amygdala/tensorflow-workshop/blob/master/workshop_sections/getting_started/xor/README.md
/// </summary>
public class NeuralNetXorEager : SciSharpExample, IExample
{
    public int num_steps = 10000;

    private NDArray data;
    private NDArray label;

    public ExampleConfig InitConfig()
        => Config = new ExampleConfig
        {
            Name = "NN XOR in Eager Mode",
            Enabled = true
        };

    public bool Run()
    {
        PrepareData();
        var loss_value = RunEagerMode();
        return loss_value < 0.0628;
    }

    private float RunEagerMode()
    {
        var learning_rate = 0.01f;
        var num_hidden = 8;
        var display_step = 1000;
        var stddev = 1 / Math.Sqrt(2);
        var features = tf.constant(data);
        var labels = tf.constant(label);

        var hidden_weights = tf.Variable(tf.random.truncated_normal((2, num_hidden), seed: 1, stddev: (float)stddev));

        // Shape [4, num_hidden]
        var hidden_activations = tf.nn.relu(tf.matmul(features, hidden_weights));

        var output_weights = tf.Variable(tf.truncated_normal(
            (num_hidden, 1),
            seed: 17,
            stddev: (float)(1 / Math.Sqrt(num_hidden))
        ));

        var optimizer = keras.optimizers.SGD(learning_rate);

        // Run training for the given number of steps.
        foreach (var step in range(1, num_steps + 1))
        {
            using var g = tf.GradientTape();

            // Shape [4, 1]
            var logits = tf.matmul(hidden_activations, output_weights);

            // Shape [4]
            var predictions = tf.tanh(tf.squeeze(logits));
            var loss = tf.reduce_mean(tf.square(predictions - tf.cast(labels, tf.float32)), name: "loss");

            // should stop recording
            // Compute gradients.
            var gradients = g.gradient(loss, output_weights);

            // Update W and b following gradients.
            optimizer.apply_gradients((gradients, output_weights));

            if (step % display_step == 0)
            {
                print($"step: {step}, loss: {loss.numpy()}");
            }
        }
        return 0;
    }

    public override void PrepareData()
    {
        data = new float[,]
        {
            {1, 0 },
            {1, 1 },
            {0, 0 },
            {0, 1 }
        };

        label = new float[,]
        {
            {1 },
            {0 },
            {0 },
            {1 }
        };
    }
}
