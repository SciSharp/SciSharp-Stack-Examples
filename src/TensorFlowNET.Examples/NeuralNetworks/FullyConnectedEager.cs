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
using System.Linq;
using Tensorflow;
using Tensorflow.Keras.Optimizers;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Build a 2-hidden layers fully connected neural network (a.k.a multilayer perceptron) with TensorFlow v2.
    /// This example is using a low-level approach to better understand all mechanics behind building neural networks and the training process.
    /// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/neural_network_raw.ipynb
    /// </summary>
    public class FullyConnectedEager : SciSharpExample, IExample
    {
        int num_classes = 10; // total classes (0-9 digits).
        int num_features = 784; // data features (img shape: 28*28).

        // Training parameters.
        float learning_rate = 0.001f;
        int training_steps = 1000;
        int batch_size = 256;
        int display_step = 100;

        // Network parameters.
        int n_hidden_1 = 128; // 1st layer number of neurons.
        int n_hidden_2 = 256; // 2nd layer number of neurons.

        IDatasetV2 train_data;
        NDArray x_test, y_test, x_train, y_train;
        IVariableV1 h1, h2, wout, b1, b2, bout;
        float accuracy_test = 0f;

        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "Fully Connected Neural Network (Eager)",
                Enabled = true,
                IsImportingGraph = false,
                Priority = 11
            };

        public bool Run()
        {
            tf.enable_eager_execution();

            PrepareData();

            // Store layers weight & bias
            // A random value generator to initialize weights.
            var random_normal = tf.initializers.random_normal_initializer();
            h1 = tf.Variable(random_normal.Apply(new InitializerArgs((num_features, n_hidden_1))));
            h2 = tf.Variable(random_normal.Apply(new InitializerArgs((n_hidden_1, n_hidden_2))));
            wout = tf.Variable(random_normal.Apply(new InitializerArgs((n_hidden_2, num_classes))));
            b1 = tf.Variable(tf.zeros(n_hidden_1));
            b2 = tf.Variable(tf.zeros(n_hidden_2));
            bout = tf.Variable(tf.zeros(num_classes));
            var trainable_variables = new IVariableV1[] { h1, h2, wout, b1, b2, bout };

            // Stochastic gradient descent optimizer.
            var optimizer = keras.optimizers.SGD(learning_rate);

            // Run training for the given number of steps.
            foreach (var (step, (batch_x, batch_y)) in enumerate(train_data, 1))
            {
                // Run the optimization to update W and b values.
                run_optimization(optimizer, batch_x, batch_y, trainable_variables);

                if (step % display_step == 0)
                {
                    var pred = neural_net(batch_x);
                    var loss = cross_entropy(pred, batch_y);
                    var acc = accuracy(pred, batch_y);
                    print($"step: {step}, loss: {(float)loss}, accuracy: {(float)acc}");
                }
            }

            // Test model on validation set.
            {
                var pred = neural_net(x_test);
                accuracy_test = (float)accuracy(pred, y_test);
                print($"Test Accuracy: {accuracy_test}");
            }

            return accuracy_test > 0.85;
        }

        void run_optimization(OptimizerV2 optimizer, Tensor x, Tensor y, IVariableV1[] trainable_variables)
        {
            using var g = tf.GradientTape();
            var pred = neural_net(x);
            var loss = cross_entropy(pred, y);

            // Compute gradients.
            var gradients = g.gradient(loss, trainable_variables);

            // Update W and b following gradients.
            optimizer.apply_gradients(zip(gradients, trainable_variables.Select(x => x as ResourceVariable)));
        }

        Tensor cross_entropy(Tensor y_pred, Tensor y_true)
        {
            // Encode label to a one hot vector.
            y_true = tf.one_hot(y_true, depth: num_classes);
            // Clip prediction values to avoid log(0) error.
            y_pred = tf.clip_by_value(y_pred, 1e-9f, 1.0f);
            // Compute cross-entropy.
            return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)));
        }

        Tensor accuracy(Tensor y_pred, Tensor y_true)
        {
            // Predicted class is the index of highest score in prediction vector (i.e. argmax).
            var correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64));
            return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis: -1);
        }

        /// <summary>
        /// Create model
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        Tensor neural_net(Tensor x)
        {
            // Hidden fully connected layer with 128 neurons.
            var layer_1 = tf.add(tf.matmul(x, h1.AsTensor()), b1.AsTensor());
            // Apply sigmoid to layer_1 output for non-linearity.
            layer_1 = tf.nn.sigmoid(layer_1);

            // Hidden fully connected layer with 256 neurons.
            var layer_2 = tf.add(tf.matmul(layer_1, h2.AsTensor()), b2.AsTensor());
            // Apply sigmoid to layer_2 output for non-linearity.
            layer_2 = tf.nn.sigmoid(layer_2);

            // Output fully connected layer with a neuron for each class.
            var out_layer = tf.matmul(layer_2, wout.AsTensor()) + bout.AsTensor();
            // Apply softmax to normalize the logits to a probability distribution.
            return tf.nn.softmax(out_layer);
        }

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
    }
}
