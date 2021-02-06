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
    /// Build a convolutional neural network with TensorFlow v2.
    /// This example is using a low-level approach to better understand all mechanics behind building convolutional neural networks and the training process.
    /// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/convolutional_network_raw.ipynb
    /// </summary>
    public class DigitRecognitionCnnEager : SciSharpExample, IExample
    {
        // MNIST dataset parameters.
        int num_classes = 10; // total classes (0-9 digits).

        // Training parameters.
        float learning_rate = 0.001f;
        int training_steps = 100;
        int batch_size = 32;
        int display_step = 10;

        // Network parameters.
        int conv1_filters = 32; // number of filters for 1st conv layer.
        int conv2_filters = 64; // number of filters for 2nd conv layer.
        int fc1_units = 1024; // number of neurons for 1st fully-connected layer.

        float accuracy_test = 0.0f;

        IDatasetV2 train_data;
        NDArray x_test, y_test, x_train, y_train;
        IVariableV1 wc1, wc2, wd1, wout;
        IVariableV1 bc1, bc2, bd1, bout;

        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "MNIST CNN (Eager)",
                Enabled = true,
                IsImportingGraph = false,
                Priority = 16
            };

        public bool Run()
        {
            tf.enable_eager_execution();

            PrepareData();

            // Store layers weight & bias

            // A random value generator to initialize weights.
            var random_normal = tf.initializers.random_normal_initializer();

            // Conv Layer 1: 5x5 conv, 1 input, 32 filters (MNIST has 1 color channel only).
            wc1 = tf.Variable(random_normal.Apply(new InitializerArgs((5, 5, 1, conv1_filters))));
            // Conv Layer 2: 5x5 conv, 32 inputs, 64 filters.
            wc2 = tf.Variable(random_normal.Apply(new InitializerArgs((5, 5, conv1_filters, conv2_filters))));
            // FC Layer 1: 7*7*64 inputs, 1024 units.
            wd1 = tf.Variable(random_normal.Apply(new InitializerArgs((7 * 7 * 64, fc1_units))));
            // FC Out Layer: 1024 inputs, 10 units (total number of classes)
            wout = tf.Variable(random_normal.Apply(new InitializerArgs((fc1_units, num_classes))));

            bc1 = tf.Variable(tf.zeros(conv1_filters));
            bc2 = tf.Variable(tf.zeros(conv2_filters));
            bd1 = tf.Variable(tf.zeros(fc1_units));
            bout = tf.Variable(tf.zeros(num_classes));

            // ADAM optimizer. 
            var optimizer = keras.optimizers.Adam(learning_rate);

            // Run training for the given number of steps.
            foreach (var (step, (batch_x, batch_y)) in enumerate(train_data, 1))
            {
                // Run the optimization to update W and b values.
                run_optimization(optimizer, batch_x, batch_y);

                if (step % display_step == 0)
                {
                    var pred = conv_net(batch_x);
                    var loss = cross_entropy(pred, batch_y);
                    var acc = accuracy(pred, batch_y);
                    print($"step: {step}, loss: {(float)loss}, accuracy: {(float)acc}");
                }
            }

            // Test model on validation set.
            {
                x_test = x_test["::100"];
                y_test = y_test["::100"];
                var pred = conv_net(x_test);
                accuracy_test = (float)accuracy(pred, y_test);
                print($"Test Accuracy: {accuracy_test}");
            }

            return accuracy_test >= 0.90;
        }

        void run_optimization(OptimizerV2 optimizer, Tensor x, Tensor y)
        {
            using var g = tf.GradientTape();
            var pred = conv_net(x);
            var loss = cross_entropy(pred, y);

            // Compute gradients.
            var trainable_variables = new IVariableV1[] { wc1, wc2, wd1, wout, bc1, bc2, bd1, bout };
            var gradients = g.gradient(loss, trainable_variables);

            // Update W and b following gradients.
            optimizer.apply_gradients(zip(gradients, trainable_variables.Select(x => x as ResourceVariable)));
        }

        Tensor conv2d(Tensor x, IVariableV1 W, IVariableV1 b, int strides = 1)
        {
            x = tf.nn.conv2d(x, W, new int[] { 1, strides, strides, 1 }, padding: "SAME");
            x = tf.nn.bias_add(x, b);
            return tf.nn.relu(x);
        }

        /// <summary>
        /// MaxPool2D wrapper.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="k"></param>
        /// <returns></returns>
        Tensor maxpool2d(Tensor x, int k = 2)
        {
            return tf.nn.max_pool(x, ksize: new[] { 1, k, k, 1 }, strides: new[] { 1, k, k, 1 }, padding: "SAME");
        }

        Tensor conv_net(Tensor x)
        {
            // Input shape: [-1, 28, 28, 1]. A batch of 28x28x1 (grayscale) images.
            x = tf.reshape(x, (-1, 28, 28, 1));

            // Convolution Layer. Output shape: [-1, 28, 28, 32].
            var conv1 = conv2d(x, wc1, bc1);

            // Max Pooling (down-sampling). Output shape: [-1, 14, 14, 32].
            conv1 = maxpool2d(conv1, k: 2);

            // Convolution Layer. Output shape: [-1, 14, 14, 64].
            var conv2 = conv2d(conv1, wc2, bc2);

            // Max Pooling (down-sampling). Output shape: [-1, 7, 7, 64].
            conv2 = maxpool2d(conv2, k: 2);

            // Reshape conv2 output to fit fully connected layer input, Output shape: [-1, 7*7*64].
            var fc1 = tf.reshape(conv2, (-1, wd1.shape.dims[0]));

            // Fully connected layer, Output shape: [-1, 1024].
            fc1 = tf.add(tf.matmul(fc1, wd1.AsTensor()), bd1.AsTensor());
            // Apply ReLU to fc1 output for non-linearity.
            fc1 = tf.nn.relu(fc1);

            // Fully connected layer, Output shape: [-1, 10].
            var output = tf.add(tf.matmul(fc1, wout.AsTensor()), bout.AsTensor());
            // Apply softmax to normalize the logits to a probability distribution.
            return tf.nn.softmax(output);
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

        public override void PrepareData()
        {
            ((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data();
            // Convert to float32.
            // (x_train, x_test) = (np.array(x_train, np.float32), np.array(x_test, np.float32));
            // Normalize images value from [0, 255] to [0, 1].
            (x_train, x_test) = (x_train / 255.0f, x_test / 255.0f);

            train_data = tf.data.Dataset.from_tensor_slices(x_train, y_train);
            train_data = train_data.repeat()
                .shuffle(5000)
                .batch(batch_size)
                .prefetch(1)
                .take(training_steps);
        }
    }
}
