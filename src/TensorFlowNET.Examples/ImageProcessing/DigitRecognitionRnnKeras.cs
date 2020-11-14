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
    /// Build a recurrent neural network (LSTM) with TensorFlow 2.0.
    /// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/recurrent_network.ipynb
    /// </summary>
    public class DigitRecognitionRnnKeras : SciSharpExample, IExample
    {
        float accuracy = 0f;
        int display_step = 100;
        int batch_size = 32;
        int training_steps = 1000;
        LSTMModel lstm_net;
        IDatasetV2 train_data;
        NDArray x_test, y_test, x_train, y_train;

        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "MNIST RNN (Keras)",
                Enabled = false,
                IsImportingGraph = false,
                Priority = 21
            };

        public bool Run()
        {
            tf.enable_eager_execution();

            PrepareData();
            BuildModel();
            Train();
            Test();

            return accuracy > 0.95;
        }

        /// <summary>
        /// Build LSTM model.
        /// </summary>
        public override void BuildModel()
        {
            lstm_net = new LSTMModel(new LSTMModelArgs
            {
                NumUnits = 32,
                NumClasses = 10,
                LearningRate = 0.001f
            });
        }

        public override void Train()
        {
            // Run training for the given number of steps.
            foreach (var (step, (batch_x, batch_y)) in enumerate(train_data, 1))
            {
                // Run the optimization to update W and b values.
                lstm_net.Optimize(batch_x, batch_y);

                if (step % display_step == 0)
                {
                    var pred = lstm_net.Predict(batch_x);
                    var loss = lstm_net.CrossEntropyLoss(pred, batch_y);
                    var acc = lstm_net.Accuracy(pred, batch_y);
                    print($"step: {step}, loss: {(float)loss}, accuracy: {(float)acc}");
                }
            }
        }

        public override void Test()
        {

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

    internal class LSTMModel : Model
    {
        OptimizerV2 optimizer;
        Layer lstm;
        Layer output;

        public LSTMModel(LSTMModelArgs args)
            : base(args)
        {
            optimizer = keras.optimizers.Adam(args.LearningRate);

            var layers = keras.layers;
            lstm = layers.LSTM(args.NumUnits);
            output = layers.Dense(args.NumClasses);
        }

        protected override Tensors Call(Tensors inputs, Tensor state = null, bool is_training = false)
        {
            // LSTM layer.
            inputs = lstm.Apply(inputs);
            // Output layer (num_classes).
            inputs = output.Apply(inputs);
            if (!is_training)
                inputs = tf.nn.softmax(inputs);
            return inputs;
        }

        /// <summary>
        /// Optimization process.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        public void Optimize(Tensor x, Tensor y)
        {
            using var g = tf.GradientTape();
            // Forward pass.
            var pred = Apply(x, is_training: true);
            // Compute loss.
            var loss = CrossEntropyLoss(pred, y);

            // Compute gradients.
            var gradients = g.gradient(loss, trainable_variables);

            // Update W and b following gradients.
            optimizer.apply_gradients(zip(gradients, trainable_variables.Select(x => x as ResourceVariable)));
        }

        /// <summary>
        /// Cross-Entropy Loss.
        /// Note that this will apply 'softmax' to the logits.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public Tensor CrossEntropyLoss(Tensor x, Tensor y)
        {
            // Convert labels to int 64 for tf cross-entropy function.
            y = tf.cast(y, tf.int64);
            // Apply softmax to logits and compute cross-entropy.
            var loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels: y, logits: x);
            // Average loss across the batch.
            return tf.reduce_mean(loss);
        }

        /// <summary>
        /// Accuracy metric.
        /// </summary>
        /// <param name="yPred"></param>
        /// <param name="yTrue"></param>
        /// <returns></returns>
        public Tensor Accuracy(Tensor yPred, Tensor yTrue)
        {
            // Predicted class is the index of highest score in prediction vector (i.e. argmax).
            var correct_prediction = tf.equal(tf.argmax(yPred, 1), tf.cast(yTrue, tf.int64));
            return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis: -1);
        }

        public Tensor Predict(Tensor x)
        {
            return Apply(x, is_training: true);
        }
    }

    internal class LSTMModelArgs : ModelArgs
    {
        public int NumUnits { get; set; }
        public int NumClasses { get; set; }
        public float LearningRate { get; set; }
    }
}
