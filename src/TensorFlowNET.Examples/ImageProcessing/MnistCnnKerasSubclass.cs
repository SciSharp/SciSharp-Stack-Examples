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
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Optimizers;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Build a convolutional neural network with TensorFlow v2.
    /// This example is using a low-level approach to better understand all mechanics behind building convolutional neural networks and the training process.
    /// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/convolutional_network.ipynb
    /// </summary>
    public class DigitRecognitionCnnKeras : SciSharpExample, IExample
    {
        // MNIST dataset parameters.
        int num_classes = 10;

        // Training parameters.
        float learning_rate = 0.001f;
        int training_steps = 100;
        int batch_size = 32;
        int display_step = 10;

        float accuracy_test = 0.0f;

        IDatasetV2 train_data;
        NDArray x_test, y_test, x_train, y_train;

        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "MNIST CNN (Keras Subclass)",
                Enabled = true,
                IsImportingGraph = false,
                Priority = 17
            };

        public bool Run()
        {
            tf.enable_eager_execution();

            PrepareData();

            // Build neural network model.
            var conv_net = new ConvNet(new ConvNetArgs
            {
                NumClasses = num_classes
            });

            // ADAM optimizer. 
            var optimizer = keras.optimizers.Adam(learning_rate);

            // Run training for the given number of steps.
            foreach (var (step, (batch_x, batch_y)) in enumerate(train_data, 1))
            {
                // Run the optimization to update W and b values.
                run_optimization(conv_net, optimizer, batch_x, batch_y);

                if (step % display_step == 0)
                {
                    var pred = conv_net.Apply(batch_x);
                    var loss = cross_entropy_loss(pred, batch_y);
                    var acc = accuracy(pred, batch_y);
                    print($"step: {step}, loss: {(float)loss}, accuracy: {(float)acc}");
                }
            }

            // Test model on validation set.
            {
                x_test = x_test["::100"];
                y_test = y_test["::100"];
                var pred = conv_net.Apply(x_test);
                accuracy_test = (float)accuracy(pred, y_test);
                print($"Test Accuracy: {accuracy_test}");
            }

            return accuracy_test > 0.90;
        }

        void run_optimization(ConvNet conv_net, OptimizerV2 optimizer, Tensor x, Tensor y)
        {
            using var g = tf.GradientTape();
            var pred = conv_net.Apply(x, is_training: true);
            var loss = cross_entropy_loss(pred, y);

            // Compute gradients.
            var gradients = g.gradient(loss, conv_net.trainable_variables);

            // Update W and b following gradients.
            optimizer.apply_gradients(zip(gradients, conv_net.trainable_variables.Select(x => x as ResourceVariable)));
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

        Tensor cross_entropy_loss(Tensor x, Tensor y)
        {
            // Convert labels to int 64 for tf cross-entropy function.
            y = tf.cast(y, tf.int64);
            // Apply softmax to logits and compute cross-entropy.
            var loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels: y, logits: x);
            // Average loss across the batch.
            return tf.reduce_mean(loss);
        }

        Tensor accuracy(Tensor y_pred, Tensor y_true)
        {
            // # Predicted class is the index of highest score in prediction vector (i.e. argmax).
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

        public class ConvNet : Model
        {
            Layer conv1;
            Layer maxpool1;
            Layer conv2;
            Layer maxpool2;
            Layer flatten;
            Layer fc1;
            Layer dropout;
            Layer output;

            public ConvNet(ConvNetArgs args)
                : base(args)
            {
                var layers = keras.layers;

                // Convolution Layer with 32 filters and a kernel size of 5.
                conv1 = layers.Conv2D(32, kernel_size: 5, activation: keras.activations.Relu);

                // Max Pooling (down-sampling) with kernel size of 2 and strides of 2.
                maxpool1 = layers.MaxPooling2D(2, strides: 2);

                // Convolution Layer with 64 filters and a kernel size of 3.
                conv2 = layers.Conv2D(64, kernel_size: 3, activation: keras.activations.Relu);
                // Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
                maxpool2 = layers.MaxPooling2D(2, strides: 2);

                // Flatten the data to a 1-D vector for the fully connected layer.
                flatten = layers.Flatten();

                // Fully connected layer.
                fc1 = layers.Dense(1024);
                // Apply Dropout (if is_training is False, dropout is not applied).
                dropout = layers.Dropout(rate: 0.5f);

                // Output layer, class prediction.
                output = layers.Dense(args.NumClasses);

                StackLayers(conv1, maxpool1, conv2, maxpool2, flatten, fc1, dropout, output);
            }

            /// <summary>
            /// Set forward pass.
            /// </summary>
            /// <param name="inputs"></param>
            /// <param name="is_training"></param>
            /// <param name="state"></param>
            /// <returns></returns>
            protected override Tensors Call(Tensors inputs, Tensor state = null, bool is_training = false)
            {
                inputs = tf.reshape(inputs, (-1, 28, 28, 1));
                inputs = conv1.Apply(inputs);
                inputs = maxpool1.Apply(inputs);
                inputs = conv2.Apply(inputs);
                inputs = maxpool2.Apply(inputs);
                inputs = flatten.Apply(inputs);
                inputs = fc1.Apply(inputs);
                inputs = dropout.Apply(inputs, is_training: is_training);
                inputs = output.Apply(inputs);

                if (!is_training)
                    inputs = tf.nn.softmax(inputs);

                return inputs;
            }
        }

        public class ConvNetArgs : ModelArgs
        {
            public int NumClasses { get; set; }
        }
    }
}
