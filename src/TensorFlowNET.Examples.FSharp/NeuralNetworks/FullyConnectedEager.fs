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
open Tensorflow
open Tensorflow.Keras.Optimizers
open type Tensorflow.Binding
open type Tensorflow.KerasApi

/// Build a 2-hidden layers fully connected neural network (a.k.a multilayer perceptron) with TensorFlow v2.
/// This example is using a low-level approach to better understand all mechanics behind building neural networks and the training process.
/// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/neural_network_raw.ipynb
module FullyConnectedEager =

    let num_classes = 10 // total classes (0-9 digits).
    let num_features = 784 // data features (img shape: 28*28).

    // Training parameters.
    let learning_rate = 0.001f
    let training_steps = 1000
    let batch_size = 256
    let display_step = 100

    // Network parameters.
    let n_hidden_1 = 128 // 1st layer number of neurons.
    let n_hidden_2 = 256 // 2nd layer number of neurons.

    let private prepareData () =
        // Prepare MNIST data.
        let struct (x_train, y_train), struct (x_test, y_test) = keras.datasets.mnist.load_data().Deconstruct()
        // Flatten images to 1-D vector of 784 features (28*28).
        let struct (x_train, x_test) = (x_train.reshape(Shape (-1, num_features)), x_test.reshape(Shape (-1, num_features)))
        // Normalize images value from [0, 255] to [0, 1].
        let (x_train, x_test) = (x_train / 255f, x_test / 255f)

        // Use tf.data API to shuffle and batch data.
        let train_data = tf.data.Dataset.from_tensor_slices(x_train.asTensor, y_train.asTensor)
        let train_data =
            train_data.repeat()
                .shuffle(5000)
                .batch(batch_size)
                .prefetch(1)
                .take(training_steps)

        (x_test, y_test), train_data

    let private run () =
        tf.enable_eager_execution()

        let ((x_test, y_test), train_data) = prepareData()

        // Store layers weight & bias
        // A random value generator to initialize weights.
        let random_normal = tf.initializers.random_normal_initializer()
        let h1 = tf.Variable(random_normal.Apply(InitializerArgs(TensorShape (num_features, n_hidden_1))))
        let h2 = tf.Variable(random_normal.Apply(InitializerArgs(TensorShape (n_hidden_1, n_hidden_2))))
        let wout = tf.Variable(random_normal.Apply(InitializerArgs(TensorShape (n_hidden_2, num_classes))))
        let b1 = tf.Variable(tf.zeros(TensorShape n_hidden_1))
        let b2 = tf.Variable(tf.zeros(TensorShape n_hidden_2))
        let bout = tf.Variable(tf.zeros(TensorShape num_classes))
        let trainable_variables : ResourceVariable[] = [| h1; h2; wout; b1; b2; bout |]

        /// Create model
        let neural_net x =
            // Hidden fully connected layer with 128 neurons.
            let layer_1 = tf.add(tf.matmul(x, h1.AsTensor()), b1.AsTensor())
            // Apply sigmoid to layer_1 output for non-linearity.
            let layer_1 = tf.nn.sigmoid(layer_1)

            // Hidden fully connected layer with 256 neurons.
            let layer_2 = tf.add(tf.matmul(layer_1, h2.AsTensor()), b2.AsTensor())
            // Apply sigmoid to layer_2 output for non-linearity.
            let layer_2 = tf.nn.sigmoid(layer_2)

            // Output fully connected layer with a neuron for each class.
            let out_layer = tf.matmul(layer_2, wout.AsTensor()) + bout.AsTensor()
            // Apply softmax to normalize the logits to a probability distribution.
            tf.nn.softmax(out_layer)

        let cross_entropy y_pred y_true =
            // Encode label to a one hot vector.
            let y_true = tf.one_hot(y_true, depth = num_classes)
            // Clip prediction values to avoid log(0) error.
            let y_pred = tf.clip_by_value(y_pred, 1e-9f, 1.0f)
            // Compute cross-entropy.
            tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

        let run_optimization (optimizer : OptimizerV2) x y (trainable_variables : ResourceVariable[]) =
            use g = tf.GradientTape()
            let pred = neural_net x
            let loss = cross_entropy pred y

            // Compute gradients.
            let gradients = g.gradient(loss, trainable_variables |> Array.map (fun x -> x :> IVariableV1))

            // Update W and b following gradients.
            optimizer.apply_gradients(zip(gradients, trainable_variables))

        let accuracy y_pred y_true =
            // Predicted class is the index of highest score in prediction vector (i.e. argmax).
            let correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
            tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis = -1)

        // Stochastic gradient descent optimizer.
        let optimizer = keras.optimizers.SGD(learning_rate)

        // Run training for the given number of steps.
        for (step, (batch_x, batch_y)) in enumerate(train_data, 1) do
            // Run the optimization to update W and b values.
            run_optimization optimizer batch_x batch_y trainable_variables

            if step % display_step = 0 then
                let pred = neural_net batch_x
                let loss = cross_entropy pred batch_y
                let acc = accuracy pred batch_y
                print($"step: {step}, loss: {float32 loss}, accuracy: {float32 acc}")

        // Test model on validation set.
        let pred = neural_net x_test.asTensor
        let accuracy_test = float32 (accuracy pred y_test.asTensor)
        print($"Test Accuracy: {accuracy_test}")

        accuracy_test > 0.85f

    let Example = { Config = ExampleConfig.Create("Fully Connected Neural Network (Eager)", priority0 = 11)
                    Run = run }

