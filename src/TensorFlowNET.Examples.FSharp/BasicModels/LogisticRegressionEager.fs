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
open type Tensorflow.Binding
open type Tensorflow.KerasApi

/// A logistic regression learning algorithm example using TensorFlow library.
/// This example is using the MNIST database of handwritten digits
/// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py
module LogisticRegressionEager =

    let training_epochs = 1000
    let batch_size = 256
    let num_classes = 10 // 0 to 9 digits
    let num_features = 784 // 28*28
    let learning_rate = 0.01f
    let display_step = 50
    let mutable accuracy = 0f

    let private run () =

        tf.enable_eager_execution()

        // Prepare MNIST data.
        let struct (x_train, y_train), struct (x_test, y_test) = keras.datasets.mnist.load_data().Deconstruct()
        // Flatten images to 1-D vector of 784 features (28*28).
        let x_train, x_test = (x_train.reshape(Shape (-1, num_features)), x_test.reshape(Shape (-1, num_features)))
        // Normalize images value from [0, 255] to [0, 1].
        let x_train, x_test = (x_train / 255f, x_test / 255f)

        // Use tf.data API to shuffle and batch data.
        let train_data = tf.data.Dataset.from_tensor_slices(x_train.asTensor, y_train.asTensor)
        let train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

        // Weight of shape [784, 10], the 28*28 image features, and total number of classes.
        let W = tf.Variable(tf.ones(TensorShape (num_features, num_classes)), name = "weight")
        // Bias of shape [10], the total number of classes.
        let b = tf.Variable(tf.zeros(TensorShape num_classes), name = "bias")

        let logistic_regression = fun x -> tf.nn.softmax(tf.matmul(x, W.asTensor) + b)

        let cross_entropy = fun (y_pred, y_true) ->
            let y_true = tf.cast(y_true, TF_DataType.TF_UINT8)
            // Encode label to a one hot vector.
            let y_true = tf.one_hot(y_true, depth = num_classes)
            // Clip prediction values to avoid log(0) error.
            let y_pred = tf.clip_by_value(y_pred, 1e-9f, 1.0f)
            // Compute cross-entropy.
            tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), 1))

        let get_accuracy = fun (y_pred, y_true) ->
            // Predicted class is the index of highest score in prediction vector (i.e. argmax).
            let correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
            tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        // Stochastic gradient descent optimizer.
        let optimizer = keras.optimizers.SGD(learning_rate)

        let run_optimization = fun (x, y) ->
            // Wrap computation inside a GradientTape for automatic differentiation.
            use g = tf.GradientTape()
            let pred = logistic_regression(x)
            let loss = cross_entropy(pred, y)

            // Compute gradients.
            let gradients = g.gradient(loss, struct (W, b))

            // Update W and b following gradients.
            optimizer.apply_gradients(zip(gradients, struct (W, b)))

        let train_data = train_data.take(training_epochs)
        // Run training for the given number of steps.
        for (step, (batch_x, batch_y)) in enumerate(train_data, 1) do

            // Run the optimization to update W and b values.
            run_optimization(batch_x, batch_y)

            if step % display_step = 0 then

                let pred = logistic_regression(batch_x)
                let loss = cross_entropy(pred, batch_y)
                let acc = get_accuracy(pred, batch_y)
                print($"step: {step}, loss: {(float32)loss}, accuracy: {(float32)acc}")
                accuracy <- float32 <| acc.numpy()

        // Test model on validation set.
        let pred = logistic_regression(x_test.asTensor)
        print($"Test Accuracy: {float32 <| get_accuracy(pred, y_test.asTensor)}")

        true

    let Example = { Config = ExampleConfig.Create("Logistic Regression (Eager)", priority0 = 7)
                    Run = run }

