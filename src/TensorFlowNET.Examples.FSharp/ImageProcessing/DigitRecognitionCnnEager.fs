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

open System.Diagnostics
open System.IO

open NumSharp
open Tensorflow
open Tensorflow.Keras
open Tensorflow.Keras.ArgsDefinition
open Tensorflow.Keras.Engine
open Tensorflow.Keras.Optimizers
open type Tensorflow.Binding
open type Tensorflow.KerasApi

module DigitRecognitionCnnEager =

    // MNIST dataset parameters.
    let num_classes = 10 // total classes (0-9 digits).

    // Training parameters.
    let learning_rate = 0.001f
    let training_steps = 100
    let batch_size = 32
    let display_step = 10

    // Network parameters.
    let conv1_filters = 32 // number of filters for 1st conv layer.
    let conv2_filters = 64 // number of filters for 2nd conv layer.
    let fc1_units = 1024 // number of neurons for 1st fully-connected layer.

    let private prepareData () =
        let struct (x_train, y_train), struct (x_test, y_test) = keras.datasets.mnist.load_data().Deconstruct()
        // Convert to float32.
        // let (x_train, x_test) = np.array(x_train, np.float32), np.array(x_test, np.float32)
        // Normalize images value from [0, 255] to [0, 1].
        let x_train, x_test = x_train / 255.0f, x_test / 255.0f

        let train_data = tf.data.Dataset.from_tensor_slices(x_train.asTensor, y_train.asTensor)
        let train_data =
            train_data.repeat()
                .shuffle(5000)
                .batch(batch_size)
                .prefetch(1)
                .take(training_steps)

        train_data, (x_test, y_test)

    type Variables = {
        wc1 : ResourceVariable
        wc2 : ResourceVariable
        wd1 : ResourceVariable
        wout : ResourceVariable
        bc1 : ResourceVariable
        bc2 : ResourceVariable
        bd1 : ResourceVariable
        bout : ResourceVariable
    }

    let conv2d x W b strides =
        let x = tf.nn.conv2d(x, W, [| 1; strides; strides; 1 |], padding = "SAME")
        let x = tf.nn.bias_add(x, b)
        tf.nn.relu(x)

    /// MaxPool2D wrapper.
    let maxpool2d x k =
        tf.nn.max_pool(x, ksize = [| 1; k; k; 1 |], strides = [| 1; k; k; 1 |], padding = "SAME")

    let conv_net variables x =
        // Input shape: [-1, 28, 28, 1]. A batch of 28x28x1 (grayscale) images.
        let x = tf.reshape(x, TensorShape (-1, 28, 28, 1))

        // Convolution Layer. Output shape: [-1, 28, 28, 32].
        let conv1 = conv2d x variables.wc1 variables.bc1 1

        // Max Pooling (down-sampling). Output shape: [-1, 14, 14, 32].
        let conv1 = maxpool2d conv1 2

        // Convolution Layer. Output shape: [-1, 14, 14, 64].
        let conv2 = conv2d conv1 variables.wc2 variables.bc2 1

        // Max Pooling (down-sampling). Output shape: [-1, 7, 7, 64].
        let conv2 = maxpool2d conv2 2

        // Reshape conv2 output to fit fully connected layer input, Output shape: [-1, 7*7*64].
        let fc1 = tf.reshape(conv2, TensorShape (-1, variables.wd1.shape.dims.[0]))

        // Fully connected layer, Output shape: [-1, 1024].
        let fc1 = tf.add(tf.matmul(fc1, variables.wd1.AsTensor()), variables.bd1.AsTensor())
        // Apply ReLU to fc1 output for non-linearity.
        let fc1 = tf.nn.relu(fc1)

        // Fully connected layer, Output shape: [-1, 10].
        let output = tf.add(tf.matmul(fc1, variables.wout.AsTensor()), variables.bout.AsTensor())
        // Apply softmax to normalize the logits to a probability distribution.
        tf.nn.softmax(output)

    let cross_entropy y_pred y_true =
        // Encode label to a one hot vector.
        let y_true = tf.one_hot(y_true, depth = num_classes)
        // Clip prediction values to avoid log(0) error.
        let y_pred = tf.clip_by_value(y_pred, 1e-9f, 1.0f)
        // Compute cross-entropy.
        tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

    let run_optimization variables (optimizer : OptimizerV2) x y =
        use g = tf.GradientTape()
        let pred = conv_net variables x
        let loss = cross_entropy pred y

        // Compute gradients.
        let trainable_variables : IVariableV1[] = [|
            variables.wc1
            variables.wc2
            variables.wd1
            variables.wout
            variables.bc1
            variables.bc2
            variables.bd1
            variables.bout
        |]
        let gradients = g.gradient(loss, trainable_variables)

        // Update W and b following gradients.
        optimizer.apply_gradients(zip(gradients, trainable_variables |> Seq.map (fun x -> x :?> ResourceVariable)))

    let accuracy y_pred y_true =
        // Predicted class is the index of highest score in prediction vector (i.e. argmax).
        let correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
        tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis = -1)

    let private run () =
        tf.enable_eager_execution()

        let train_data, (x_test, y_test) = prepareData()

        // Store layers weight & bias

        // A random value generator to initialize weights.
        let random_normal = tf.initializers.random_normal_initializer()

        // Conv Layer 1: 5x5 conv, 1 input, 32 filters (MNIST has 1 color channel only).
        let wc1 = tf.Variable(random_normal.Apply(InitializerArgs(TensorShape (5, 5, 1, conv1_filters))))
        // Conv Layer 2: 5x5 conv, 32 inputs, 64 filters.
        let wc2 = tf.Variable(random_normal.Apply(InitializerArgs(TensorShape (5, 5, conv1_filters, conv2_filters))))
        // FC Layer 1: 7*7*64 inputs, 1024 units.
        let wd1 = tf.Variable(random_normal.Apply(InitializerArgs(TensorShape (7 * 7 * 64, fc1_units))))
        // FC Out Layer: 1024 inputs, 10 units (total number of classes)
        let wout = tf.Variable(random_normal.Apply( InitializerArgs(TensorShape (fc1_units, num_classes))))

        let bc1 = tf.Variable(tf.zeros(TensorShape conv1_filters))
        let bc2 = tf.Variable(tf.zeros(TensorShape conv2_filters))
        let bd1 = tf.Variable(tf.zeros(TensorShape fc1_units))
        let bout = tf.Variable(tf.zeros(TensorShape num_classes))

        let variables = {
            wc1 = wc1
            wc2 = wc2
            wd1 = wd1
            wout = wout
            bc1 = bc1
            bc2 = bc2
            bd1 = bd1
            bout = bout
        }

        // ADAM optimizer. 
        let optimizer = keras.optimizers.Adam(learning_rate)

        // Run training for the given number of steps.
        for step, (batch_x, batch_y) in enumerate(train_data, 1) do
            // Run the optimization to update W and b values.
            run_optimization variables optimizer batch_x batch_y

            if step % display_step = 0 then
                let pred = conv_net variables batch_x
                let loss = cross_entropy pred batch_y
                let acc = accuracy pred batch_y
                print($"step: {step}, loss: {(float32)loss}, accuracy: {(float32)acc}")

        // Test model on validation set.
        let x_test = x_test.["::100"]
        let y_test = y_test.["::100"]
        let pred = conv_net variables x_test.asTensor
        let accuracy_test = float32 <| accuracy pred y_test.asTensor
        print($"Test Accuracy: {accuracy_test}")

        accuracy_test >= 0.90f

    let Example = { Config = ExampleConfig.Create("MNIST CNN (Eager)", priority = 16 )
                    Run = run }

