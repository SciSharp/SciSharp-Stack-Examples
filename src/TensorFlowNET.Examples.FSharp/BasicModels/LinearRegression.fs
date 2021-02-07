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

open NumSharp
open System
open Tensorflow
open type Tensorflow.Binding

module LinearRegression =

    let training_epochs = 1000

    // Parameters
    let learning_rate = 0.01f
    let display_step = 50

    let private prepareData() =
        let train_X = np.array(3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 7.59f, 2.167f,
                               7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f)
        let train_Y = np.array(1.7f, 2.76f, 2.09f, 3.19f, 1.694f, 1.573f, 3.366f, 2.596f, 2.53f, 1.221f,
                               2.827f, 3.465f, 1.65f, 2.904f, 2.42f, 2.94f, 1.3f)
        let n_samples = train_X.shape.[0]
        (train_X, train_Y, n_samples)

    let private run() =
        tf.compat.v1.disable_eager_execution()

        // Training data
        let train_X, train_Y, n_samples = prepareData()

        // tf Graph Input
        let X = tf.placeholder(tf.float32)
        let Y = tf.placeholder(tf.float32)

        // Set model weights 
        // We can set a fixed init value in order to debug
        // let rnd1 = rng.randn<float>()
        // let rnd2 = rng.randn<float>()
        let W = tf.Variable(-0.06f, name = "weight")
        let b = tf.Variable(-0.73f, name = "bias")

        // Construct a linear model
        let pred = tf.add(tf.multiply(X, W), b)

        // Mean squared error
        let cost = tf.reduce_sum(tf.pow(pred - Y, 2.0f)) / (2.0f * float32 n_samples)

        // Gradient descent
        // Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
        let optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

        // Initialize the variables (i.e. assign their default value)
        let init = tf.global_variables_initializer()

        // Start training
        use sess = tf.Session()
        // Run the initializer
        sess.run(init)

        let items array = Array.map FeedItem array

        // Fit all training data
        for epoch in 1 .. training_epochs do
            for x, y in zip<float>(train_X, train_Y) do
                sess.run(optimizer, items [| X, x; Y, y |])

            // Display logs per epoch step
            if epoch % display_step = 0 then
                let c = sess.run(cost, items [| X, train_X; Y, train_Y |])
                printfn $"Epoch: %i{epoch} cost=%s{c.ToString()} W={sess.run(W).[0]} b={sess.run(b).[0]}"

        printfn "Optimization Finished!"
        let training_cost = sess.run(cost, items [| X, train_X; Y, train_Y |])
        printfn $"Training cost=%s{training_cost.ToString()} W={sess.run(W).[0]} b={sess.run(b).[0]}"

        // Testing example
        let test_X = np.array(6.83f, 4.668f, 8.9f, 7.91f, 5.7f, 8.7f, 3.1f, 2.1f)
        let test_Y = np.array(1.84f, 2.273f, 3.2f, 2.831f, 2.92f, 3.24f, 1.35f, 1.03f)
        printfn "Testing... (Mean square loss Comparison)"
        let testing_cost = sess.run(tf.reduce_sum(tf.pow(pred - Y, 2.0f)) / (2.0f * float32 test_X.shape.[0]),
                                                  items [| X, test_X; Y, test_Y |])
        printfn $"Testing cost=%A{testing_cost}"
        let diff = Math.Abs(training_cost.GetAtIndex<float32>(0) - testing_cost.GetAtIndex<float32>(0))
        printfn $"Absolute mean square loss difference: {diff}"

        diff < 0.01f

    let Example =
        { SciSharpExample.Config = ExampleConfig.Create ("Linear Regression (Graph)", priority0 = 4)
          Run = run
        }

