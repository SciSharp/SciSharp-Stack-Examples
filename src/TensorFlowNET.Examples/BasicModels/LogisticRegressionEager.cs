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
using System.Diagnostics;
using System.IO;
using Tensorflow;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// A logistic regression learning algorithm example using TensorFlow library.
    /// This example is using the MNIST database of handwritten digits
    /// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py
    /// </summary>
    public class LogisticRegressionEager : SciSharpExample, IExample
    {
        int training_epochs = 1000;
        int? train_size = null;
        int validation_size = 5000;
        int? test_size = null;
        int batch_size = 256;
        int num_classes = 10; // 0 to 9 digits
        int num_features = 784; // 28*28
        float learning_rate = 0.01f;
        int display_step = 50;
        float accuracy = 0f;

        Datasets<MnistDataSet> mnist;

        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "Logistic Regression (Eager)",
                Enabled = true,
                IsImportingGraph = false,
                Priority = 7
            };

        public bool Run()
        {
            tf.enable_eager_execution();

            RunEagerMode();

            return accuracy > 0.8;
        }

        public void RunEagerMode()
        {
            // Prepare MNIST data.
            var ((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data();
            // Flatten images to 1-D vector of 784 features (28*28).
            (x_train, x_test) = (x_train.reshape((-1, num_features)), x_test.reshape((-1, num_features)));
            // Normalize images value from [0, 255] to [0, 1].
            (x_train, x_test) = (x_train / 255f, x_test / 255f);

            // Use tf.data API to shuffle and batch data.
            var train_data = tf.data.Dataset.from_tensor_slices(x_train, y_train);
            train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1);

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
            var optimizer = keras.optimizers.SGD(learning_rate);

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

            train_data = train_data.take(training_epochs);
            // Run training for the given number of steps.
            foreach (var (step, (batch_x, batch_y)) in enumerate(train_data, 1))
            {
                // Run the optimization to update W and b values.
                run_optimization(batch_x, batch_y);

                if (step % display_step == 0)
                {
                    var pred = logistic_regression(batch_x);
                    var loss = cross_entropy(pred, batch_y);
                    var acc = accuracy(pred, batch_y);
                    print($"step: {step}, loss: {(float)loss}, accuracy: {(float)acc}");
                    this.accuracy = acc.numpy();
                }
            }

            // Test model on validation set.
            {
                var pred = logistic_regression(x_test);
                print($"Test Accuracy: {(float)accuracy(pred, y_test)}");
            }
        }

        public override void Train()
        {
            // tf Graph Input
            var x = tf.placeholder(tf.float32, new TensorShape(-1, 784)); // mnist data image of shape 28*28=784
            var y = tf.placeholder(tf.float32, new TensorShape(-1, 10)); // 0-9 digits recognition => 10 classes

            // Set model weights
            var W = tf.Variable(tf.zeros(new Shape(784, 10)));
            var b = tf.Variable(tf.zeros(new Shape(10)));

            // Construct model
            var pred = tf.nn.softmax(tf.matmul(x, W) + b); // Softmax

            // Minimize error using cross entropy
            var cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices: 1));

            // Gradient Descent
            var optimizer = tf.train.GradientDescentOptimizer(learning_rate);
            var loss = optimizer.minimize(cost);

            // Initialize the variables (i.e. assign their default value)
            var init = tf.global_variables_initializer();

            var total_batch = mnist.Train.NumOfExamples / batch_size;

            var sw = new Stopwatch();

            using (var sess = tf.Session())
            {
                // Run the initializer
                sess.run(init);

                // Training cycle
                foreach (var epoch in range(training_epochs))
                {
                    sw.Start();
                    var avg_cost = 0.0f;

                    // Loop over all batches
                    foreach (var i in range(total_batch))
                    {
                        var start = i * batch_size;
                        var end = (i + 1) * batch_size;
                        var (batch_xs, batch_ys) = mnist.GetNextBatch(mnist.Train.Data, mnist.Train.Labels, start, end);
                        // Run optimization op (backprop) and cost op (to get loss value)
                        (_, float c) = sess.run((loss, cost),
                            (x, batch_xs),
                            (y, batch_ys));

                        // Compute average loss
                        avg_cost += c / total_batch;
                    }

                    sw.Stop();

                    // Display logs per epoch step
                    if ((epoch + 1) % display_step == 0)
                        print($"Epoch: {(epoch + 1):D4} Cost: {avg_cost:G9} Elapse: {sw.ElapsedMilliseconds}ms");

                    sw.Reset();
                }

                print("Optimization Finished!");
                // SaveModel(sess);

                // Test model
                var correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1));
                // Calculate accuracy
                var acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32));
                accuracy = acc.eval(sess, (x, mnist.Test.Data), (y, mnist.Test.Labels));
                print($"Accuracy: {acc:F4}");
            }
        }

        public void SaveModel(Session sess)
        {
            var saver = tf.train.Saver();
            var save_path = saver.save(sess, ".resources/logistic_regression/model.ckpt");
            tf.train.write_graph(sess.graph, ".resources/logistic_regression", "model.pbtxt", as_text: true);

            FreezeGraph.freeze_graph(input_graph: ".resources/logistic_regression/model.pbtxt",
                              input_saver: "",
                              input_binary: false,
                              input_checkpoint: ".resources/logistic_regression/model.ckpt",
                              output_node_names: "Softmax",
                              restore_op_name: "save/restore_all",
                              filename_tensor_name: "save/Const:0",
                              output_graph: ".resources/logistic_regression/model.pb",
                              clear_devices: true,
                              initializer_nodes: "");
        }

        public override void Predict()
        {
            var graph = new Graph().as_default();
            using (var sess = tf.Session(graph))
            {
                graph.Import(Path.Join(".resources/logistic_regression", "model.pb"));

                // restoring the model
                // var saver = tf.train.import_meta_graph("logistic_regression/tensorflowModel.ckpt.meta");
                // saver.restore(sess, tf.train.latest_checkpoint('logistic_regression'));
                var pred = graph.OperationByName("Softmax");
                var output = pred.outputs[0];
                var x = graph.OperationByName("Placeholder");
                var input = x.outputs[0];

                // predict
                var (batch_xs, batch_ys) = mnist.Train.GetNextBatch(10);
                var results = sess.run(output, new FeedItem(input, batch_xs[np.arange(1)]));

                if (results[0].argmax() == (batch_ys[0] as NDArray).argmax())
                    print("predicted OK!");
                else
                    throw new ValueError("predict error, should be 90% accuracy");
            }
        }
    }
}
