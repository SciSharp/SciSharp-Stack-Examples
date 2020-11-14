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
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Build a convolutional neural network with TensorFlow v2.
    /// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/neural_network.ipynb
    /// </summary>
    public class FullyConnectedKeras : SciSharpExample, IExample
    {
        // MNIST dataset parameters.
        int num_classes = 10; // 0 to 9 digits
        int num_features = 784; // 28*28

        // Training parameters.
        float learning_rate = 0.1f;
        int display_step = 100;
        int batch_size = 256;
        int training_steps = 1000;

        float accuracy;
        IDatasetV2 train_data;
        NDArray x_test, y_test, x_train, y_train;

        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "Fully Connected Neural Network (Keras)",
                Enabled = true,
                IsImportingGraph = false,
                Priority = 12
            };

        public override void PrepareData()
        {
            // Prepare MNIST data.
            ((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data();
            // Flatten images to 1-D vector of 784 features (28*28).
            (x_train, x_test) = (x_train.reshape((-1, num_features)), x_test.reshape((-1, num_features)));
            // Normalize images value from [0, 255] to [0, 1].
            (x_train, x_test) = (x_train / 255f, x_test / 255f);

            // Use tf.data API to shuffle and batch data.
            train_data = tf.data.Dataset.from_tensor_slices(x_train, y_train);
            train_data = train_data.repeat()
                .shuffle(5000)
                .batch(batch_size)
                .prefetch(1)
                .take(training_steps);
        }

        public bool Run()
        {
            tf.enable_eager_execution();

            PrepareData();

            // Build neural network model.
            var neural_net = new NeuralNet(new NeuralNetArgs
            {
                NumClasses = num_classes,
                NeuronOfHidden1 = 128,
                Activation1 = keras.activations.Relu,
                NeuronOfHidden2 = 256,
                Activation2 = keras.activations.Relu
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
            var optimizer = keras.optimizers.SGD(learning_rate);

            // Optimization process.
            Action<Tensor, Tensor> run_optimization = (x, y) =>
            {
                // Wrap computation inside a GradientTape for automatic differentiation.
                using var g = tf.GradientTape();
                // Forward pass.
                var pred = neural_net.Apply(x, is_training: true);
                var loss = cross_entropy_loss(pred, y);

                // Compute gradients.
                var gradients = g.gradient(loss, neural_net.trainable_variables);

                // Update W and b following gradients.
                optimizer.apply_gradients(zip(gradients, neural_net.trainable_variables.Select(x => x as ResourceVariable)));
            };


            // Run training for the given number of steps.
            foreach (var (step, (batch_x, batch_y)) in enumerate(train_data, 1))
            {
                // Run the optimization to update W and b values.
                run_optimization(batch_x, batch_y);

                if (step % display_step == 0)
                {
                    var pred = neural_net.Apply(batch_x, is_training: true);
                    var loss = cross_entropy_loss(pred, batch_y);
                    var acc = accuracy(pred, batch_y);
                    print($"step: {step}, loss: {(float)loss}, accuracy: {(float)acc}");
                }
            }

            // Test model on validation set.
            {
                var pred = neural_net.Apply(x_test, is_training: false);
                this.accuracy = (float)accuracy(pred, y_test);
                print($"Test Accuracy: {this.accuracy}");
            }

            return this.accuracy > 0.92f;
        }

        public class NeuralNet : Model
        {
            Layer fc1;
            Layer fc2;
            Layer output;

            public NeuralNet(NeuralNetArgs args) :
                base(args)
            {
                var layers = keras.layers;

                // First fully-connected hidden layer.
                fc1 = layers.Dense(args.NeuronOfHidden1, activation: args.Activation1);

                // Second fully-connected hidden layer.
                fc2 = layers.Dense(args.NeuronOfHidden2, activation: args.Activation2);

                output = layers.Dense(args.NumClasses);

                StackLayers(fc1, fc2, output);
            }

            // Set forward pass.
            protected override Tensors Call(Tensors inputs, Tensor state = null, bool is_training = false)
            {
                inputs = fc1.Apply(inputs);
                inputs = fc2.Apply(inputs);
                inputs = output.Apply(inputs);
                if (!is_training)
                    inputs = tf.nn.softmax(inputs);
                return inputs;
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
            public Activation Activation1 { get; set; }

            /// <summary>
            /// 2nd layer number of neurons.
            /// </summary>
            public int NeuronOfHidden2 { get; set; }
            public Activation Activation2 { get; set; }

            public int NumClasses { get; set; }
        }
    }
}
