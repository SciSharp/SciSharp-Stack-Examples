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

open NumSharp
open Tensorflow
open type Tensorflow.Binding
open type Tensorflow.KerasApi

/// A nearest neighbor learning algorithm example
/// This example is using the MNIST database of handwritten digits
/// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/nearest_neighbor.py
module NearestNeighbor =
    let TrainSize : Nullable<int> = Nullable()
    let ValidationSize = 5000
    let TestSize : Nullable<int> = Nullable()

    let prepareData () =
        let mnist =
            MnistModelLoader.LoadAsync(
                ".resources/mnist",
                oneHot = true,
                trainSize = TrainSize,
                validationSize = ValidationSize,
                testSize = TestSize,
                showProgressInConsole = true).Result
        // In this example, we limit mnist data
        let struct (Xtr, Ytr) = mnist.Train.GetNextBatch(if TrainSize = Nullable() then 5000 else TrainSize.Value / 100) // 5000 for training (nn candidates)
        let struct (Xte, Yte) = mnist.Test.GetNextBatch(if TestSize = Nullable() then 200 else TestSize.Value / 100) // 200 for testing
        mnist, (Xtr, Ytr), (Xte, Yte)

    let private run () =
        tf.compat.v1.disable_eager_execution()
        // tf Graph Input
        let xtr = tf.placeholder(tf.float32, new TensorShape(-1, 784))
        let xte = tf.placeholder(tf.float32, new TensorShape(784))

        // Nearest Neighbor calculation using L1 Distance
        // Calculate L1 Distance
        let distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices = 1)
        // Prediction: Get min distance index (Nearest neighbor)
        let pred = tf.arg_min(distance, 0)

        let mutable accuracy = 0f
        // Initialize the variables (i.e. assign their default value)
        let init = tf.global_variables_initializer()
        use sess = tf.Session()
        // Run the initializer
        sess.run(init)

        let mnist, (Xtr, Ytr), (Xte, Yte) = prepareData()

        let items array = Array.map FeedItem array

        for i in range(Xte.shape.[0]) do

            // Get nearest neighbor
            let nn_index = int64 <| sess.run(pred, items [| xtr, Xtr; xte, Xte.[i] |])
            // Get nearest neighbor class label and compare it to its true label
            let index = int nn_index

            if i % 10 = 0 || i = 0 then
                print($"Test {i} Prediction: {np.argmax(Ytr.[index])} True Class: {np.argmax(Yte.[i])}")

            // Calculate accuracy
            if np.argmax(Ytr.[index]) = np.argmax(Yte.[i]) then
                accuracy <- accuracy + 1f / float32 Xte.shape.[0]

        print($"Accuracy: {accuracy}")

        accuracy > 0.8f

    let Example = { Config = ExampleConfig.Create("Nearest Neighbor", priority0 = 8)
                    Run = run }
