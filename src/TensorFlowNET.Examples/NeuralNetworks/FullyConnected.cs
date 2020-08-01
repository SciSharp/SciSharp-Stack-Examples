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
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Operations.Activation;
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
            if (tf.Context.executing_eagerly())
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
            int training_steps = 10;

            // Prepare MNIST data.
            var ((x_train, y_train), (x_test, y_test)) = tf.keras.datasets.mnist.load_data();
            // Flatten images to 1-D vector of 784 features (28*28).
            (x_train, x_test) = (x_train.reshape((-1, num_features)), x_test.reshape((-1, num_features)));
            // Normalize images value from [0, 255] to [0, 1].
            (x_train, x_test) = (x_train / 255f, x_test / 255f);

            // Use tf.data API to shuffle and batch data.
            var train_data = tf.data.Dataset.from_tensor_slices(x_train, y_train);
            train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1);

            // Build neural network model.
            var neural_net = new NeuralNet(new NeuralNetArgs
            {
                NeuronOfHidden1 = 128,
                Activation1 = tf.nn.relu(),
                NeuronOfHiddenHidden2 = 256,
                Activation2 = tf.nn.relu()
            });

            // Cross-Entropy Loss.
            // Note that this will apply 'softmax' to the logits.
            Func<Tensor, Tensor, Tensor> cross_entropy_loss = (x, y) =>
            {
                // Convert labels to int 64 for tf cross-entropy function.
                y = tf.cast(y, tf.int64);
                // Apply softmax to logits and compute cross-entropy.
                var loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels: y, logits: x);
                // Average loss across the batch.
                return tf.reduce_mean(loss);
            };

            // Accuracy metric.
            Func<Tensor, Tensor, Tensor> accuracy = (y_pred, y_true) =>
            {
                // Predicted class is the index of highest score in prediction vector (i.e. argmax).
                var correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64));
                return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis: -1);
            };

            // Stochastic gradient descent optimizer.
            var optimizer = tf.optimizers.SGD(learning_rate);

            // Optimization process.
            Action<Tensor, Tensor> run_optimization = (x, y) =>
            {
                // Wrap computation inside a GradientTape for automatic differentiation.
                using var g = tf.GradientTape();
                // Forward pass.
                var pred = neural_net.Apply(new[] { x }, is_training: true);
                var loss = cross_entropy_loss(pred, y);

                // Compute gradients.
                var gradients = g.gradient(loss, neural_net.TrainableVariables);
            };

            train_data = train_data.take(training_steps);
            // Run training for the given number of steps.
            foreach (var (step, (batch_x, batch_y)) in enumerate(train_data, 1))
            {
                // Run the optimization to update W and b values.
                run_optimization(batch_x, batch_y);

                if (step % display_step == 0)
                {
                    var pred = neural_net.Apply(new[] { batch_x }, is_training: true);
                    var loss = cross_entropy_loss(pred, batch_y);
                    var acc = accuracy(pred, batch_y);
                    print($"step: {step}, loss: {(float)loss}, accuracy: {(float)acc}");
                    this.accuracy = acc.numpy();
                }
            }

            // Test model on validation set.
            {
                var pred = neural_net.Apply(new[] { (Tensor)x_test }, is_training: false);
            }
        }

        public class NeuralNet : Model
        {
            Layer fc1;
            Layer fc2;
            Layer output;

            public NeuralNet(NeuralNetArgs args) : 
                base(args)
            {
                // First fully-connected hidden layer.
                //fc1 = tf.keras.layers.Dense(args.NeuronOfHidden1, activation: args.Activation1);

                // Second fully-connected hidden layer.
                //fc2 = tf.keras.layers.Dense(args.NeuronOfHiddenHidden2, activation: args.Activation2);

                output = tf.keras.layers.Dense(args.NumClasses);
            }

            // Set forward pass.
            protected override Tensor[] call(Tensor[] inputs, bool is_training = false, Tensor state = null)
            {
                var x = fc1.Apply(inputs);
                throw new NotImplementedException("");
            }
        }

        /// <summary>
        /// Network parameters.
        /// </summary>
        public class NeuralNetArgs : ModelArgs
        {
            /// <summary>
            /// 1st layer number of neurons.
            /// </summary>
            public int NeuronOfHidden1 { get; set; }
            public IActivation Activation1 { get; set; }

            /// <summary>
            /// 2nd layer number of neurons.
            /// </summary>
            public int NeuronOfHiddenHidden2 { get; set; }
            public IActivation Activation2 { get; set; }

            public int NumClasses { get; set; }
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
