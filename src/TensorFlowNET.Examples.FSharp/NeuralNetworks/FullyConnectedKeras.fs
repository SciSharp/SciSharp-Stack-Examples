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
open Tensorflow.Keras
open Tensorflow.Keras.ArgsDefinition
open Tensorflow.Keras.Engine
open type Tensorflow.Binding
open type Tensorflow.KerasApi

/// Build a convolutional neural network with TensorFlow v2.
/// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/neural_network.ipynb
module FullyConnectedKeras =

    // MNIST dataset parameters.
    let num_classes = 10 // 0 to 9 digits
    let num_features = 784 // 28*28

    // Training parameters.
    let learning_rate = 0.1f
    let display_step = 100
    let batch_size = 256
    let training_steps = 1000

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

    type NeuralNetArgs = {
        /// 1st layer number of neurons.
        NeuronOfHidden1 : int
        Activation1 : Activation

        /// 2nd layer number of neurons.
        NeuronOfHidden2 : int
        Activation2 : Activation

        NumClasses : int
    }

    type NeuralNet(fc1 : Layer, fc2 : Layer, output : Layer) as this =
        inherit Model(ModelArgs())

        private new(args : NeuralNetArgs) =
            let layers = keras.layers
            // First fully-connected hidden layer.
            let fc1 = layers.Dense(args.NeuronOfHidden1, activation = args.Activation1)
            // Second fully-connected hidden layer.
            let fc2 = layers.Dense(args.NeuronOfHidden2, activation = args.Activation2)
            let output = layers.Dense(args.NumClasses)
            NeuralNet(fc1, fc2, output)

        member private x.StackLayers() =
            x.StackLayers([| fc1; fc2; output |] |> Array.map (fun l -> l :> ILayer))

        static member Create(args : NeuralNetArgs) =
            let nn = NeuralNet(args)
            nn.StackLayers()
            nn

        member x.Call(inputs : Tensors) =
            x.Call(inputs, null, false)

        // Set forward pass.
        override _.Call(inputs : Tensors, state : Tensor, is_training : bool) =
            let inputs = fc1.Apply(inputs)
            let inputs = fc2.Apply(inputs)
            let inputs = output.Apply(inputs)
            if not is_training then
                tf.nn.softmax(inputs.asTensor).asTensors
            else
                inputs

    let private run () =
        tf.enable_eager_execution()

        let ((x_test, y_test), train_data) = prepareData()

        // Build neural network model.
        let neural_net = NeuralNet.Create({
            NumClasses = num_classes
            NeuronOfHidden1 = 128
            Activation1 = keras.activations.Relu
            NeuronOfHidden2 = 256
            Activation2 = keras.activations.Relu
        })

        // Cross-Entropy Loss.
        // Note that this will apply 'softmax' to the logits.
        let cross_entropy_loss x y =
            // Convert labels to int 64 for tf cross-entropy function.
            let y = tf.cast(y, tf.int64)
            // Apply softmax to logits and compute cross-entropy.
            let loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = x)
            // Average loss across the batch.
            tf.reduce_mean(loss)

        // Accuracy metric.
        let get_accuracy y_pred y_true =
            // Predicted class is the index of highest score in prediction vector (i.e. argmax).
            let correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
            tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis = -1)

        // Stochastic gradient descent optimizer.
        let optimizer = keras.optimizers.SGD(learning_rate)

        // Optimization process.
        let run_optimization (x : Tensor) y =
            // Wrap computation inside a GradientTape for automatic differentiation.
            use g = tf.GradientTape()
            // Forward pass.
            let pred = neural_net.Apply(x.asTensors, is_training = true)
            let loss = cross_entropy_loss pred.asTensor y

            // Compute gradients.
            let gradients = g.gradient(loss, neural_net.trainable_variables)

            // Update W and b following gradients.
            optimizer.apply_gradients(zip(gradients, neural_net.trainable_variables |> Seq.map (fun x -> x :?> ResourceVariable)))

        // Run training for the given number of steps.
        for (step, (batch_x, batch_y)) in enumerate(train_data, 1) do

            // Run the optimization to update W and b values.
            run_optimization batch_x batch_y

            if step % display_step = 0 then
                let pred = neural_net.Apply(batch_x.asTensors, is_training = true)
                let loss = cross_entropy_loss pred.asTensor batch_y
                let acc = get_accuracy pred.asTensor batch_y
                print($"step: {step}, loss: {float32 loss}, accuracy: {float32 acc}")

        // Test model on validation set.
        let pred = neural_net.Apply(x_test.asTensor.asTensors, is_training = false)
        let accuracy = float32 (get_accuracy pred.asTensor y_test.asTensor)
        print($"Test Accuracy: {accuracy}")

        accuracy > 0.92f

    let Example = { Config = ExampleConfig.Create("Fully Connected Neural Network (Keras)", priority0 = 12)
                    Run = run }

