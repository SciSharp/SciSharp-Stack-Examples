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
using Tensorflow.Hub;
using static Tensorflow.Binding;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Recurrent Neural Network for handwritten digits MNIST.
    /// https://medium.com/machine-learning-algorithms/mnist-using-recurrent-neural-network-2d070a5915a2
    /// </summary>
    public class DigitRecognitionRNN : SciSharpExample, IExample
    {
        // Hyper-parameters
        int n_neurons = 128;
        float learning_rate = 0.001f;
        int batch_size = 128;
        int n_epochs = 5;

        int n_steps = 28;
        int n_inputs = 28;
        int n_outputs = 10;

        Datasets<MnistDataSet> mnist;

        Tensor X, y;
        Tensor loss, accuracy, prediction;
        Operation optimizer;

        int display_freq = 100;
        float accuracy_test = 0f;
        float loss_test = 1f;

        NDArray x_train, y_train;
        NDArray x_valid, y_valid;
        NDArray x_test, y_test;

        Session sess;

        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "MNIST RNN",
                Enabled = true,
                IsImportingGraph = false,
                Priority = 10
            };

        public bool Run()
        {
            PrepareData();
            BuildGraph();

            sess = tf.Session();

            Train();
            Test();

            return accuracy_test > 0.95;
        }

        public override Graph BuildGraph()
        {
            var graph = new Graph().as_default();

            X = tf.placeholder(tf.float32, new[] { -1, n_steps, n_inputs });
            y = tf.placeholder(tf.int32, new[] { -1 });
            var cell = tf.nn.rnn_cell.BasicRNNCell(num_units: n_neurons);
            var (output, state) = tf.nn.dynamic_rnn(cell, X, dtype: tf.float32);
            var logits = tf.layers.dense(state, n_outputs);
            var cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels: y, logits: logits);
            loss = tf.reduce_mean(cross_entropy);
            var adam = tf.train.AdamOptimizer(learning_rate: learning_rate);
            optimizer = adam.minimize(loss);
            prediction = tf.nn.in_top_k(logits, y, 1);
            accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32));

            return graph;
        }

        public override void Train()
        {
            float loss_val = 100.0f;
            float accuracy_val = 0f;

            // Number of training iterations in each epoch
            var n_batches = y_train.shape[0] / batch_size;

            var init = tf.global_variables_initializer();
            sess.run(init);

            foreach (var epoch in range(n_epochs))
            {
                print($"Training epoch: {epoch + 1}");
                // Randomly shuffle the training data at the beginning of each epoch 
                (x_train, y_train) = mnist.Randomize(x_train, y_train);

                foreach (var iteration in range(n_batches))
                {
                    var start = iteration * batch_size;
                    var end = (iteration + 1) * batch_size;
                    var (X_train, y_batch) = mnist.GetNextBatch(x_train, y_train, start, end);
                    X_train = X_train.reshape(-1, n_steps, n_inputs);
                    y_batch = np.argmax(y_batch, axis: 1);
                    // Run optimization op (backprop)
                    sess.run(optimizer, new FeedItem(X, X_train), new FeedItem(y, y_batch));

                    if (iteration % display_freq == 0)
                    {
                        // Calculate and display the batch loss and accuracy
                        (loss_val, accuracy_val) = sess.run((loss, accuracy), (X, X_train), (y, y_batch));
                        print($"iter {iteration.ToString("000")}: Loss={loss_val.ToString("0.0000")}, Training Accuracy={accuracy_val.ToString("P")}");
                    }
                }

                // Run validation after every epoch
                (loss_val, accuracy_val) = sess.run((loss, accuracy), (X, x_valid), (y, y_valid));

                print("---------------------------------------------------------");
                print($"Epoch: {epoch + 1}, validation loss: {loss_val.ToString("0.0000")}, validation accuracy: {accuracy_val.ToString("P")}");
                print("---------------------------------------------------------");
            }
        }

        public override void Test()
        {
            var result = sess.run(new[] { loss, accuracy }, new FeedItem(X, x_test), new FeedItem(y, y_test));
            loss_test = result[0];
            accuracy_test = result[1];
            print("---------------------------------------------------------");
            print($"Test loss: {loss_test.ToString("0.0000")}, test accuracy: {accuracy_test.ToString("P")}");
            print("---------------------------------------------------------");
        }

        public override void PrepareData()
        {
            mnist = MnistModelLoader.LoadAsync(".resources/mnist", oneHot: true, showProgressInConsole: true).Result;
            (x_train, y_train) = (mnist.Train.Data, mnist.Train.Labels);
            (x_valid, y_valid) = (mnist.Validation.Data, mnist.Validation.Labels);
            (x_test, y_test) = (mnist.Test.Data, mnist.Test.Labels);

            y_valid = np.argmax(y_valid, axis: 1);
            x_valid = x_valid.reshape(-1, n_steps, n_inputs);
            y_test = np.argmax(y_test, axis: 1);
            x_test = x_test.reshape(-1, n_steps, n_inputs);

            print("Size of:");
            print($"- Training-set:\t\t{len(mnist.Train.Data)}");
            print($"- Validation-set:\t{len(mnist.Validation.Data)}");
            print($"- Test-set:\t\t{len(mnist.Test.Data)}");
        }
    }
}
