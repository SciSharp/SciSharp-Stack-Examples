(*****************************************************************************
Copyright 2021 The TensorFlow.NET Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
******************************************************************************)

namespace TensorFlowNET.Examples.FSharp

open System
open System.Diagnostics
open System.IO

open NumSharp
open Tensorflow
open type Tensorflow.Binding

/// A logistic regression learning algorithm example using TensorFlow library.
/// This example is using the MNIST database of handwritten digits
/// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py
module LogisticRegression =

    let training_epochs = 10
    let train_size : Nullable<int> = Nullable()
    let validation_size = 5000
    let test_size : Nullable<int> = Nullable()
    let batch_size = 100

    let learning_rate = 0.01
    let display_step = 1

    let private prepareData() =
        MnistModelLoader.LoadAsync(
            ".resources/mnist",
            oneHot = true,
            trainSize = train_size,
            validationSize = validation_size,
            testSize = test_size,
            showProgressInConsole = true).Result

    let private saveModel sess =
        let saver = tf.train.Saver()
        saver.save(sess, ".resources/logistic_regression/model.ckpt") |> printfn "%s"
        tf.train.write_graph(sess.graph, ".resources/logistic_regression", "model.pbtxt", as_text = true) |> printfn "%s"

        FreezeGraph.freeze_graph(input_graph = ".resources/logistic_regression/model.pbtxt",
                                 input_saver = "",
                                 input_binary = false,
                                 input_checkpoint = ".resources/logistic_regression/model.ckpt",
                                 output_node_names = "Softmax",
                                 restore_op_name = "save/restore_all",
                                 filename_tensor_name = "save/Const:0",
                                 output_graph = ".resources/logistic_regression/model.pb",
                                 clear_devices = true,
                                 initializer_nodes = "")

    let private train (mnist : Datasets<MnistDataSet>) =
        // tf Graph Input
        let x = tf.placeholder(tf.float32, new TensorShape(-1, 784)) // mnist data image of shape 28*28=784
        let y = tf.placeholder(tf.float32, new TensorShape(-1, 10)) // 0-9 digits recognition => 10 classes

        // Set model weights
        let W = tf.Variable(tf.zeros(Shape(784, 10).asTensorShape))
        let b = tf.Variable(tf.zeros(Shape(10).asTensorShape))

        // Construct model
        let pred = tf.nn.softmax(tf.matmul(x, W.asTensor) + b) // Softmax

        // Minimize error using cross entropy
        let cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices = 1))

        // Gradient Descent
        let optimizer = tf.train.GradientDescentOptimizer(float32 learning_rate).minimize(cost)

        // Initialize the variables (i.e. assign their default value)
        let init = tf.global_variables_initializer()

        let total_batch = mnist.Train.NumOfExamples / batch_size

        let sw = new Stopwatch()

        use sess = tf.Session()
        // Run the initializer
        sess.run(init)

        let items array = Array.map FeedItem array

        // Training cycle
        for epoch in range(training_epochs) do
            sw.Start()
            let mutable avg_cost = 0.0f

            // Loop over all batches
            for i in range(total_batch) do
                let start = i * batch_size
                let ``end`` = (i + 1) * batch_size
                let struct (batch_xs, batch_ys) = mnist.GetNextBatch(mnist.Train.Data, mnist.Train.Labels, start, ``end``)
                // Run optimization op (backprop) and cost op (to get loss value)
                let struct (_, c) = sess.run(struct (optimizer :> ITensorOrOperation, cost :> ITensorOrOperation), items [| x, batch_xs; y, batch_ys |])

                // Compute average loss
                avg_cost <- avg_cost + (float32 c) / float32 total_batch

            sw.Stop()

            // Display logs per epoch step
            if (epoch + 1) % display_step = 0 then
                print($"Epoch: {(epoch + 1):D4} Cost: {avg_cost:G9} Elapsed: {sw.ElapsedMilliseconds}ms")

            sw.Reset()

        print("Optimization Finished!")
        //saveModel sess

        // Test model
        let correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        // Calculate accuracy
        let acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        let accuracy = acc.eval(sess, items [| x, mnist.Test.Data; y, mnist.Test.Labels |])
        print($"Accuracy: {accuracy:F4}")
        float accuracy

    let predict (mnist : Datasets<MnistDataSet>) =
        use g = new Graph()
        let graph = g.as_default()
        use sess = tf.Session(graph)
        graph.Import(Path.Join(".resources/logistic_regression", "model.pb")) |> ignore

        // restoring the model
        // let saver = tf.train.import_meta_graph("logistic_regression/tensorflowModel.ckpt.meta")
        // saver.restore(sess, tf.train.latest_checkpoint('logistic_regression'))
        let pred = graph.OperationByName("Softmax")
        let output = pred.outputs.[0]
        let x = graph.OperationByName("Placeholder")
        let input = x.outputs.[0]

        // predict
        let struct (batch_xs, batch_ys) = mnist.Train.GetNextBatch(10)
        let results = sess.run(output, new FeedItem(input, batch_xs.[np.arange(1)]))

        if results.[0].argmax() = (batch_ys.[0]).argmax() then
            print("predicted OK!")
        else
            raise (ValueError("predict error, should be 90% accuracy"))

    let private run () =
        tf.compat.v1.disable_eager_execution()

        let mnist = prepareData()
        let accuracy = train mnist
        //predict mnist

        accuracy > 0.9

    let Example =
        { Config = ExampleConfig.Create("Logistic Regression (Graph)", priority0 = 6)
          Run = run
        }

