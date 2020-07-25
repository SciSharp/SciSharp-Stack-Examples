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
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Build a 2-hidden layers fully connected neural network (a.k.a multilayer perceptron)
    /// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/neural_network.ipynb
    /// </summary>
    public class FullyConnected : SciSharpExample, IExample
    {
        Tensor input = null;
        Tensor x_inputs_data = null;
        Tensor y_inputs_data = null;
        Tensor accuracy = null;
        Tensor y_true = null;
        Tensor loss_op = null;
        Operation train_op = null;

        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "Fully Connected Neural Network",
                Enabled = true,
                IsImportingGraph = false,
                Priority = 13
            };

        public override Graph BuildGraph()
        {
            var g = tf.get_default_graph();
            
            Tensor z = null;
            
            tf_with(tf.variable_scope("placeholder"), delegate
            {
                input = tf.placeholder(tf.float32, shape: (-1, 1024));
                y_true = tf.placeholder(tf.int32, shape: (-1, 1));
            });

            tf_with(tf.variable_scope("FullyConnected"), delegate
            {
                var w = tf.compat.v1.get_variable("w", shape: (1024, 1024), initializer: tf.random_normal_initializer(stddev: 0.1f));
                var b = tf.compat.v1.get_variable("b", shape: 1024, initializer: tf.constant_initializer(0.1));
                z = tf.matmul(input, w.AsTensor()) + b.AsTensor();
                var y = tf.nn.relu(z);

                var w2 = tf.compat.v1.get_variable("w2", shape: (1024, 1), initializer: tf.random_normal_initializer(stddev: 0.1f));
                var b2 = tf.compat.v1.get_variable("b2", shape: 1, initializer: tf.constant_initializer(0.1));
                z = tf.matmul(y, w2.AsTensor()) + b2.AsTensor();
            });

            tf_with(tf.variable_scope("Loss"), delegate
            {
                var losses = tf.nn.sigmoid_cross_entropy_with_logits(tf.cast(y_true, tf.float32), z);
                loss_op = tf.reduce_mean(losses);
            });

            tf_with(tf.variable_scope("Accuracy"), delegate
            {
                var y_pred = tf.cast(z > 0, tf.int32);
                accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32));
                // accuracy = tf.Print(accuracy, data =[accuracy], message = "accuracy:")
            });

            // We add the training operation, ...
            var adam = tf.train.AdamOptimizer(0.01f);
            train_op = adam.minimize(loss_op, name: "train_op");

            return g;
        }

        public override void PrepareData()
        {
            // batches of 128 samples, each containing 1024 data points
            x_inputs_data = tf.random.normal(new[] { 128, 1024 }, mean: 0, stddev: 1);
            // We will try to predict this law:
            // predict 1 if the sum of the elements is positive and 0 otherwise
            y_inputs_data = tf.cast(tf.reduce_sum(x_inputs_data, axis: 1, keepdims: true) > 0, tf.int32);
        }

        public bool Run()
        {
            if (tf.context.executing_eagerly())
                RunEagerMode();
            else
            {
                PrepareData();
                BuildGraph();
                Train();
            }

            return true;
        }

        public void RunEagerMode()
        {
            // MNIST dataset parameters.
            int num_classes = 10; // 0 to 9 digits
            int num_features = 784; // 28*28
            // Training parameters.
            float learning_rate = 0.1f;
            int display_step = 100;
            int batch_size = 256;
            int training_steps = 2000;

            // Network parameters.
            int n_hidden_1 = 128; // 1st layer number of neurons.
            int n_hidden_2 = 256; // 2nd layer number of neurons.

            // Prepare MNIST data.
            var ((x_train, y_train), (x_test, y_test)) = tf.keras.datasets.mnist.load_data();
            // Flatten images to 1-D vector of 784 features (28*28).
            (x_train, x_test) = (x_train.reshape((-1, num_features)), x_test.reshape((-1, num_features)));
            // Normalize images value from [0, 255] to [0, 1].
            (x_train, x_test) = (x_train / 255f, x_test / 255f);

            // Use tf.data API to shuffle and batch data.
            var train_data = tf.data.Dataset.from_tensor_slices(x_train, y_train);

            // Weight of shape [784, 10], the 28*28 image features, and total number of classes.
            var W = tf.Variable(tf.ones((num_features, num_classes)), name: "weight");
            // Bias of shape [10], the total number of classes.
            var b = tf.Variable(tf.zeros(num_classes), name: "bias");

            Func<Tensor, Tensor> logistic_regression = x
                => tf.nn.softmax(tf.matmul(x, W) + b);

            Func<Tensor, Tensor, Tensor> cross_entropy = (y_pred, y_true) =>
            {
                y_true = tf.cast(y_true, TF_DataType.TF_UINT8);
                // Encode label to a one hot vector.
                y_true = tf.one_hot(y_true, depth: num_classes);
                // Clip prediction values to avoid log(0) error.
                y_pred = tf.clip_by_value(y_pred, 1e-9f, 1.0f);
                // Compute cross-entropy.
                return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), 1));
            };

            Func<Tensor, Tensor, Tensor> accuracy = (y_pred, y_true) =>
            {
                // Predicted class is the index of highest score in prediction vector (i.e. argmax).
                var correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64));
                return tf.reduce_mean(tf.cast(correct_prediction, tf.float32));
            };

            // Stochastic gradient descent optimizer.
            var optimizer = tf.optimizers.SGD(learning_rate);

            Action<Tensor, Tensor> run_optimization = (x, y) =>
            {
                // Wrap computation inside a GradientTape for automatic differentiation.
                using var g = tf.GradientTape();
                var pred = logistic_regression(x);
                var loss = cross_entropy(pred, y);

                // Compute gradients.
                var gradients = g.gradient(loss, (W, b));

                // Update W and b following gradients.
                optimizer.apply_gradients(zip(gradients, (W, b)));
            };

            // Run training for the given number of steps.
            foreach (var step in range(1, training_steps))
            {
                var start = (step - 1) * batch_size;
                var end = step * batch_size;
                var (batch_x, batch_y) = (x_test, y_test); //mnistV1.GetNextBatch(mnistV1.Train.Data, mnistV1.Train.Labels, start, end);
                // Run the optimization to update W and b values.
                var x_tensor = tf.constant(batch_x);
                var y_tensor = tf.constant(np.argmax(batch_y, 1));
                run_optimization(x_tensor, y_tensor);

                if (step % display_step == 0)
                {
                    var pred = logistic_regression(x_tensor);
                    var loss = cross_entropy(pred, y_tensor);
                    var acc = accuracy(pred, y_tensor);
                    print($"step: {step}, loss: {loss.numpy()}, accuracy: {acc.numpy()}");
                }
            }
        }

        public override void Train()
        {
            var sw = new Stopwatch();
            sw.Start();

            var config = new ConfigProto
            {
                IntraOpParallelismThreads = 1,
                InterOpParallelismThreads = 1,
                LogDevicePlacement = true
            };

            using (var sess = tf.Session(config))
            {
                // init variables
                sess.run(tf.global_variables_initializer());

                // check the accuracy before training
                var (x_input, y_input) = sess.run((x_inputs_data, y_inputs_data));
                sess.run(accuracy, (input, x_input), (y_true, y_input));

                // training
                foreach (var i in range(5000))
                {
                    // by sampling some input data (fetching)
                    (x_input, y_input) = sess.run((x_inputs_data, y_inputs_data));
                    var (_, loss) = sess.run((train_op, loss_op), (input, x_input), (y_true, y_input));

                    // We regularly check the loss
                    if (i % 500 == 0)
                        print($"iter:{i} - loss:{loss}");
                }

                // Finally, we check our final accuracy
                (x_input, y_input) = sess.run((x_inputs_data, y_inputs_data));
                sess.run(accuracy, (input, x_input), (y_true, y_input));
            }

            print($"Time taken: {sw.Elapsed.TotalSeconds}s");
        }
    }
}
