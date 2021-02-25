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

open Tensorflow
open Tensorflow.Keras
open Tensorflow.Keras.ArgsDefinition
open Tensorflow.Keras.Engine
open Tensorflow.Keras.Optimizers
open type Tensorflow.Binding
open type Tensorflow.KerasApi

module MnistCnnKerasSubclass =
    // MNIST dataset parameters.
    let num_classes = 10

    // Training parameters.
    let learning_rate = 0.001f
    let training_steps = 100
    let batch_size = 32
    let display_step = 10

    let accuracy_test = 0.0f

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

    type ConvNetArgs() =
        inherit ModelArgs()
        member val NumClasses = 0 with get, set

    type ConvNet(args, conv1, maxpool1, conv2, maxpool2, flatten, fc1, dropout, output) =
        inherit Model(args)
        
        private new(args : ConvNetArgs) =
            let layers = keras.layers

            // Convolution Layer with 32 filters and a kernel size of 5.
            let conv1 = layers.Conv2D(32, kernel_size = TensorShape 5, activation = keras.activations.Relu) :> Layer

            // Max Pooling (down-sampling) with kernel size of 2 and strides of 2.
            let maxpool1 = layers.MaxPooling2D(TensorShape 2, strides = TensorShape 2) :> Layer

            // Convolution Layer with 64 filters and a kernel size of 3.
            let conv2 = layers.Conv2D(64, kernel_size = TensorShape 3, activation = keras.activations.Relu)  :> Layer
            // Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
            let maxpool2 = layers.MaxPooling2D(TensorShape 2, strides = TensorShape 2) :> Layer

            // Flatten the data to a 1-D vector for the fully connected layer.
            let flatten = layers.Flatten() :> Layer

            // Fully connected layer.
            let fc1 = layers.Dense(1024) :> Layer
            // Apply Dropout (if is_training is False, dropout is not applied).
            let dropout = layers.Dropout(rate = 0.5f) :> Layer

            // Output layer, class prediction.
            let output = layers.Dense(args.NumClasses) :> Layer

            ConvNet(args, conv1, maxpool1, conv2, maxpool2, flatten, fc1, dropout, output)

        member private x.StackLayers() =
            let layers =
                [| conv1; maxpool1; conv2; maxpool2; flatten; fc1; dropout; output |]
                |> Array.map (fun l -> l :> ILayer)
            x.StackLayers(layers)

        static member Create(args : ConvNetArgs) =
            let cnn = ConvNet(args)
            cnn.StackLayers()
            cnn

        //let state = defaultArg state null
        //let is_training = defaultArg is_training false

        override x.Call(inputs, state, is_training) =
            let inputs = tf.reshape(inputs.asTensor, TensorShape (-1, 28, 28, 1)).asTensors
            let inputs = conv1.Apply(inputs)
            let inputs = maxpool1.Apply(inputs)
            let inputs = conv2.Apply(inputs)
            let inputs = maxpool2.Apply(inputs)
            let inputs = flatten.Apply(inputs)
            let inputs = fc1.Apply(inputs)
            let inputs = dropout.Apply(inputs, is_training = is_training)
            let inputs = output.Apply(inputs)
            if not is_training then tf.nn.softmax(inputs.asTensor).asTensors else inputs

    let cross_entropy_loss x y =
        // Convert labels to int 64 for tf cross-entropy function.
        let y = tf.cast(y, tf.int64)
        // Apply softmax to logits and compute cross-entropy.
        let loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = x)
        // Average loss across the batch.
        tf.reduce_mean(loss)

    let run_optimization (conv_net : ConvNet) (optimizer : OptimizerV2) (x : Tensor) y =
        use g = tf.GradientTape()
        let pred = conv_net.Apply(x.asTensors, is_training = true)
        let loss = cross_entropy_loss pred.asTensor y

        // Compute gradients.
        let gradients = g.gradient(loss, conv_net.trainable_variables)

        // Update W and b following gradients.
        optimizer.apply_gradients(zip(gradients, conv_net.trainable_variables |> Seq.map (fun x -> x :?> ResourceVariable)))

    let private accuracy y_pred y_true =
        // # Predicted class is the index of highest score in prediction vector (i.e. argmax).
        let correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
        tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis = -1)

    let private run () =
        tf.enable_eager_execution()

        let train_data, (x_test, y_test) = prepareData()

        // Build neural network model.
        let args = ConvNetArgs()
        args.NumClasses <- num_classes
        let conv_net = ConvNet.Create(args)

        // ADAM optimizer. 
        let optimizer = keras.optimizers.Adam(learning_rate)

        // Run training for the given number of steps.
        for step, (batch_x, batch_y) in enumerate(train_data, 1) do

            // Run the optimization to update W and b values.
            run_optimization conv_net optimizer batch_x batch_y

            if step % display_step = 0 then
                let pred = conv_net.Apply(batch_x.asTensors)
                let loss = cross_entropy_loss pred.asTensor batch_y
                let acc = accuracy pred.asTensor batch_y
                print($"step: {step}, loss: {(float32)loss}, accuracy: {(float32)acc}")

        // Test model on validation set.
        let x_test = x_test.["::100"]
        let y_test = y_test.["::100"]
        let pred = conv_net.Apply(x_test.asTensors)
        let accuracy_test = float32 <| accuracy pred.asTensor y_test.asTensor
        print($"Test Accuracy: {accuracy_test}")

        accuracy_test > 0.90f

    let Example = { Config = ExampleConfig.Create("MNIST CNN (Keras Subclass)", priority0 = 17)
                    Run = run }

