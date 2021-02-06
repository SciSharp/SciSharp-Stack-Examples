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

using NumSharp;
using System;
using Tensorflow;
using Tensorflow.Keras.Utils;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Simple vanilla neural net solving the famous XOR problem
    /// https://github.com/amygdala/tensorflow-workshop/blob/master/workshop_sections/getting_started/xor/README.md
    /// </summary>
    public class NeuralNetXor : SciSharpExample, IExample
    {
        public int num_steps = 10000;

        private NDArray data;
        private NDArray label;

        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "NN XOR",
                Enabled = false,
                IsImportingGraph = false,
                Priority = 3
            };

        private (Operation, Tensor, Tensor) make_graph(Tensor features, Tensor labels, int num_hidden = 8)
        {
            var stddev = 1 / Math.Sqrt(2);
            var hidden_weights = tf.Variable(tf.truncated_normal(new int[] { 2, num_hidden }, seed: 1, stddev: (float)stddev));

            // Shape [4, num_hidden]
            var hidden_activations = tf.nn.relu(tf.matmul(features, hidden_weights));

            var output_weights = tf.Variable(tf.truncated_normal(
                new[] { num_hidden, 1 },
                seed: 17,
                stddev: (float)(1 / Math.Sqrt(num_hidden))
            ));

            // Shape [4, 1]
            var logits = tf.matmul(hidden_activations, output_weights);

            // Shape [4]
            var predictions = tf.tanh(tf.squeeze(logits));
            var loss = tf.reduce_mean(tf.square(predictions - tf.cast(labels, tf.float32)), name: "loss");

            var gs = tf.Variable(0, trainable: false, name: "global_step");
            var optimizer = tf.train.GradientDescentOptimizer(0.2f);
            var train_op = optimizer.minimize(loss, global_step: gs);

            return (train_op, loss, gs);
        }

        public bool Run()
        {
            tf.compat.v1.disable_eager_execution();

            PrepareData();
            float loss_value = 0;

            if (tf.Context.executing_eagerly())
                loss_value = RunEagerMode();
            else if (Config.IsImportingGraph)
                loss_value = RunWithImportedGraph();
            else
                loss_value = RunWithBuiltGraph();

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

            var optimizer = keras.optimizers.SGD(learning_rate);

            // Run training for the given number of steps.
            foreach (var step in range(1, num_steps + 1))
            {
                using var g = tf.GradientTape();

                // Shape [4, num_hidden]
                var hidden_activations = tf.nn.relu(tf.matmul(features, hidden_weights));

                var output_weights = tf.Variable(tf.truncated_normal(
                    (num_hidden, 1),
                    seed: 17,
                    stddev: (float)(1 / Math.Sqrt(num_hidden))
                ));

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

        private float RunWithImportedGraph()
        {
            var graph = tf.Graph().as_default();

            tf.train.import_meta_graph("graph/xor.meta");

            Tensor features = graph.get_operation_by_name("Placeholder");
            Tensor labels = graph.get_operation_by_name("Placeholder_1");
            Tensor loss = graph.get_operation_by_name("loss");
            Tensor train_op = graph.get_operation_by_name("train_op");
            Tensor global_step = graph.get_operation_by_name("global_step");

            var init = tf.global_variables_initializer();
            float loss_value = 0;
            // Start tf session
            using (var sess = tf.Session(graph))
            {
                sess.run(init);
                var step = 0;

                var y_ = np.array(new int[] { 1, 0, 0, 1 }, dtype: np.int32);
                while (step < num_steps)
                {
                    // original python:
                    //_, step, loss_value = sess.run(
                    //          [train_op, gs, loss],
                    //          feed_dict={features: xy, labels: y_}
                    //      )
                    (_, step, loss_value) = sess.run((train_op, global_step, loss), (features, data), (labels, y_));
                    if (step == 1 || step % 1000 == 0)
                        Console.WriteLine($"Step {step} loss: {loss_value}");
                }
                Console.WriteLine($"Final loss: {loss_value}");
            }

            return loss_value;
        }

        private float RunWithBuiltGraph()
        {
            var graph = tf.Graph().as_default();

            var features = tf.placeholder(tf.float32, new TensorShape(4, 2));
            var labels = tf.placeholder(tf.int32, new TensorShape(4));

            var (train_op, loss, gs) = make_graph(features, labels);

            var init = tf.global_variables_initializer();

            float loss_value = 0;
            // Start tf session
            using (var sess = tf.Session(graph))
            {
                sess.run(init);
                var step = 0;

                var y_ = np.array(new int[] { 1, 0, 0, 1 }, dtype: np.int32);
                while (step < num_steps)
                {
                    (_, step, loss_value) = sess.run((train_op, gs, loss), (features, data), (labels, y_));
                    if (step == 1 || step % 1000 == 0)
                        Console.WriteLine($"Step {step} loss: {loss_value}");
                }
                Console.WriteLine($"Final loss: {loss_value}");
            }

            return loss_value;
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

            if (Config.IsImportingGraph)
            {
                // download graph meta data
                string url = "https://raw.githubusercontent.com/SciSharp/TensorFlow.NET/master/graph/xor.meta";
                Web.Download(url, "graph", "xor.meta");
            }
        }
    }
}
