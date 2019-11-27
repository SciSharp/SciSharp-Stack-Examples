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
using System.Linq;
using Tensorflow;
using Tensorflow.Hub;
using Tensorflow.Operations;
using static Tensorflow.Binding;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Bi-directional Recurrent Neural Network.
    /// 
    /// To classify images using a bidirectional recurrent neural network, we consider
    /// every image row as a sequence of pixels.Because MNIST image shape is 28*28px,
    /// we will then handle 28 sequences of 28 steps for every sample.
    /// 
    /// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py
    /// </summary>
    public class DigitRecognitionLSTM : IExample
    {
        public bool Enabled { get; set; } = true;
        public bool IsImportingGraph { get; set; } = false;

        public string Name => "MNIST LSTM";

        // Training Parameters
        float learning_rate = 0.001f;
        int training_steps = 10000;
        int batch_size = 128;
        int display_step = 200;

        // Network Parameters
        int num_input = 28;
        int timesteps = 28;
        int num_hidden = 128; // hidden layer num of features
        int num_classes = 10; // MNIST total classes (0-9 digits)

        Datasets<MnistDataSet> mnist;

        Tensor X, Y;
        Tensor loss_op, accuracy, prediction;
        Operation optimizer;
        
        float accuracy_test = 0f;
        float loss_test = 1f;

        NDArray x_train, y_train;
        NDArray x_valid, y_valid;
        NDArray x_test, y_test;

        public bool Run()
        {
            PrepareData();
            BuildGraph();

            using (var sess = tf.Session())
            {
                Train(sess);
                Test(sess);
            }

            return accuracy_test > 0.75;
        }

        public Graph BuildGraph()
        {
            var graph = new Graph().as_default();

            X = tf.placeholder(tf.float32, (-1, timesteps, num_input));
            Y = tf.placeholder(tf.float32, (-1, num_classes));

            // Hidden layer weights => 2*n_hidden because of forward + backward cells
            var weights = tf.Variable(tf.random_normal((2 * num_hidden, num_classes)));
            var biases = tf.Variable(tf.random_normal(num_classes));

            // Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
            var x = tf.unstack(X, timesteps, 1);

            // Define lstm cells with tensorflow
            // Forward direction cell
            var lstm_fw_cell = new BasicLstmCell(num_hidden, forget_bias: 1.0f);
            // Backward direction cell
            var lstm_bw_cell = new BasicLstmCell(num_hidden, forget_bias: 1.0f);

            // Get lstm cell output
            var (outputs, _, _) = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype: tf.float32);

            // Linear activation, using rnn inner loop last output
            var logits = tf.matmul(outputs.Last(), weights) + biases;
            prediction = tf.nn.softmax(logits);

            // Define loss and optimizer
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits: logits, labels: Y));
            var optimizer1 = tf.train.GradientDescentOptimizer(learning_rate: learning_rate);
            optimizer = optimizer1.minimize(loss_op);

            // Evaluate model (with test logits, for dropout to be disabled)
            var correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1));
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32));

            return graph;
        }

        public void Train(Session sess)
        {
            float loss_val = 100.0f;
            float accuracy_val = 0f;

            // Number of training iterations in each epoch
            var n_batches = y_train.shape[0] / batch_size;

            var init = tf.global_variables_initializer();
            sess.run(init);

            // Randomly shuffle the training data at the beginning of each epoch 
            (x_train, y_train) = mnist.Randomize(x_train, y_train);

            foreach (var step in range(1, training_steps + 1))
            {
                var (batch_x, batch_y) = mnist.Train.GetNextBatch(batch_size);
                // Reshape data to get 28 seq of 28 elements
                batch_x = batch_x.reshape(batch_size, timesteps, num_input);
                // Run optimization op (backprop)
                sess.run(optimizer, (X, batch_x), (Y, batch_y));

                if (step % display_step == 0 || step == 1)
                {
                    // Calculate batch loss and accuracy
                    (loss_val, accuracy_val) = sess.run((loss_op, accuracy), (X, batch_x), (Y, batch_y));
                    print($"Step {step}: Minibatch Loss=={loss_val.ToString("0.0000")}, Training Accuracy={accuracy_val.ToString("0.000")}");
                }
            }

            // Calculate accuracy for 128 mnist test images
            // (loss_val, accuracy_val) = sess.run((loss, accuracy), (X, x_valid), (Y, y_valid));
        }

        public void Test(Session sess)
        {
            var result = sess.run(new[] { loss_op, accuracy }, new FeedItem(X, x_test), new FeedItem(Y, y_test));
            loss_test = result[0];
            accuracy_test = result[1];
            print("---------------------------------------------------------");
            print($"Test loss: {loss_test.ToString("0.0000")}, test accuracy: {accuracy_test.ToString("P")}");
            print("---------------------------------------------------------");
        }

        public void PrepareData()
        {
            mnist = MnistModelLoader.LoadAsync(".resources/mnist", oneHot: true, showProgressInConsole: true).Result;
            (x_train, y_train) = (mnist.Train.Data, mnist.Train.Labels);
            (x_valid, y_valid) = (mnist.Validation.Data, mnist.Validation.Labels);
            (x_test, y_test) = (mnist.Test.Data, mnist.Test.Labels);

            y_valid = np.argmax(y_valid, axis: 1);
            x_valid = x_valid.reshape(-1, training_steps, num_input);
            y_test = np.argmax(y_test, axis: 1);
            x_test = x_test.reshape(-1, training_steps, num_input);

            print("Size of:");
            print($"- Training-set:\t\t{len(mnist.Train.Data)}");
            print($"- Validation-set:\t{len(mnist.Validation.Data)}");
            print($"- Test-set:\t\t{len(mnist.Test.Data)}");
        }

        public Graph ImportGraph() => throw new NotImplementedException();

        public void Predict(Session sess) => throw new NotImplementedException();
    }
}
