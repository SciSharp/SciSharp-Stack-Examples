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
open Tensorflow.Keras.Engine
open Tensorflow.Keras.Layers
open type Tensorflow.Binding
open type Tensorflow.KerasApi

module MnistFnnKerasFunctional =
    let prepareData () =
        let (x_train, y_train, x_test, y_test) = keras.datasets.mnist.load_data().Deconstruct()
        let x_train = x_train.reshape(60000, 784) / 255f
        let x_test = x_test.reshape(10000, 784) / 255f
        (x_train, y_train, x_test, y_test)

    let buildModel () =
        // input layer
        let inputs = keras.Input(shape = TensorShape 784)

        let layers = LayersApi()

        // 1st dense layer
        let outputs = layers.Dense(64, activation = keras.activations.Relu).Apply(inputs.asTensors)

        // 2nd dense layer
        let outputs = layers.Dense(64, activation = keras.activations.Relu).Apply(outputs)

        // output layer
        let outputs = layers.Dense(10).Apply(outputs)

        // build keras model
        let model = keras.Model(inputs.asTensors, outputs, name = "mnist_model")
        // show model summary
        model.summary()

        // compile keras model into tensorflow's static graph
        model.compile(
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits = true),
            optimizer = keras.optimizers.RMSprop(),
            metrics = [| "accuracy" |])

        model

    let private train (x_train : NDArray, y_train) (x_test : NDArray, y_test) (model : Functional) =
        // train model by feeding data and labels.
        model.fit(x_train, y_train, batch_size = 64, epochs = 2, validation_split = 0.2f)

        // evluate the model
        model.evaluate(x_test, y_test, verbose = 2)

        // save and serialize model
        model.save("mnist_model")

        // recreate the exact same model purely from the file:
        // model = keras.models.load_model("path_to_my_model")

    let private run () =
        tf.enable_eager_execution()

        let (x_train, y_train, x_test, y_test) = prepareData()
        let model = buildModel()
        train (x_train, y_train) (x_test, y_test) model

        true

    let Example = { Config = ExampleConfig.Create("MNIST FNN (Keras Functional)", priority = 17)
                    Run = run }
