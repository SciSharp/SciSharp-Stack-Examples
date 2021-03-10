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
open System.Diagnostics
open System.Drawing
open System.IO
open Tensorflow
open Tensorflow.Keras.Engine
open Tensorflow.Keras.Utils
open type Tensorflow.Binding
open type Tensorflow.KerasApi

/// <summary>
/// A toy ResNet model
/// https://keras.io/guides/functional_api/
/// </summary>
module CIFAR10_CNN =
    let layers = keras.layers

    let private buildModel () =
        let inputs = keras.Input(shape = TensorShape (32, 32, 3), name = "img")
        let x = layers.Conv2D(32, TensorShape 3, activation = "relu").Apply(inputs.asTensors)
        let x = layers.Conv2D(64, TensorShape 3, activation = "relu").Apply(x)

        // let x = layers.BatchNormalization().Apply(x)
        let block_1_output = layers.MaxPooling2D(TensorShape 3).Apply(x)

        let x = layers.Conv2D(64, TensorShape 3, activation = "relu", padding = "same").Apply(block_1_output)
        let x = layers.Conv2D(64, TensorShape 3, activation = "relu", padding = "same").Apply(x)
        let block_2_output = layers.Add().Apply(new Tensors(x.asTensor, block_1_output.asTensor))

        let x = layers.Conv2D(64, TensorShape 3, activation = "relu", padding = "same").Apply(block_2_output)
        let x = layers.Conv2D(64, TensorShape 3, activation = "relu", padding = "same").Apply(x)
        let block_3_output = layers.Add().Apply(new Tensors(x.asTensor, block_2_output.asTensor))

        let x = layers.Conv2D(64, TensorShape 3, activation = "relu").Apply(block_3_output)
        let x = layers.GlobalAveragePooling2D().Apply(x)
        let x = layers.Dense(256, activation = "relu").Apply(x)
        let x = layers.Dropout(0.5f).Apply(x)
        let outputs = layers.Dense(10).Apply(x)

        let model = keras.Model(inputs.asTensors, outputs, name = "toy_resnet")
        model.summary()

        model.compile(
            optimizer = keras.optimizers.RMSprop(1e-3f),
            loss = keras.losses.CategoricalCrossentropy(from_logits = true),
            metrics = [| "acc" |])

        model

    let private prepareData () =
        let (struct (x_train, y_train), struct (x_test, y_test)) = keras.datasets.cifar10.load_data().Deconstruct()

        let x_train = x_train / 255.0f
        let x_test = x_test / 255.0f

        let y_train = np_utils.to_categorical(y_train, 10)
        let y_test = np_utils.to_categorical(y_test, 10)

        (x_train, y_train), (x_test, y_test)

    let private train (model : Functional) (x_train : NDArray, y_train : NDArray) =
        model.fit(x_train.[Slice(0, 2000)], y_train.[Slice(0, 2000)], 
                  batch_size = 64, 
                  epochs = 3, 
                  validation_split = 0.2f)

    let private run () =
        tf.enable_eager_execution()
        
        let model = buildModel()
        let (x_train, y_train), (x_test, y_test) = prepareData()
        train model (x_train, y_train)
        
        true

    let Example = { Config = ExampleConfig.Create("Toy ResNet")
                    Run = run }

