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

open System.IO
open Tensorflow
open Tensorflow.Keras
open Tensorflow.Keras.Engine
open Tensorflow.Keras.Utils
open type Tensorflow.Binding
open type Tensorflow.KerasApi

module ImageClassificationKeras =
    let batch_size = 32
    let epochs = 10
    let img_dim = TensorShape (180, 180)

    let private prepareData () =
        let fileName = "flower_photos.tgz"
        let url = $"https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
        let data_dir = Path.Combine(Path.GetTempPath(), "flower_photos")
        Web.Download(url, data_dir, fileName) |> ignore
        Compress.ExtractTGZ(Path.Join(data_dir, fileName), data_dir)
        let data_dir = Path.Combine(data_dir, "flower_photos")

        // convert to tensor
        let train_ds =
            keras.preprocessing.image_dataset_from_directory(
                data_dir,
                validation_split = 0.2f,
                subset = "training",
                seed = 123,
                image_size = img_dim,
                batch_size = batch_size)

        let val_ds =
            keras.preprocessing.image_dataset_from_directory(
                data_dir,
                validation_split = 0.2f,
                subset = "validation",
                seed = 123,
                image_size = img_dim,
                batch_size = batch_size)

        let train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = -1)
        let val_ds = val_ds.cache().prefetch(buffer_size = -1)

        for img, label in train_ds do
            print($"images: {img.TensorShape}")
            print($"labels: {label.numpy()}")

        train_ds, val_ds
    
    let private buildModel () =
        let num_classes = 5
        // let normalization_layer = tf.keras.layers.Rescaling(1.0f / 255)
        let layers = keras.layers
        let model = keras.Sequential(ResizeArray<ILayer>(seq {
            layers.Rescaling(1.0f / 255f, input_shape = TensorShape (img_dim.dims.[0], img_dim.dims.[1], 3)) :> ILayer;
            layers.Conv2D(16, TensorShape 3, padding = "same", activation = keras.activations.Relu);
            layers.MaxPooling2D();
            (*layers.Conv2D(32, 3, padding = "same", activation = "relu");
            layers.MaxPooling2D();
            layers.Conv2D(64, 3, padding = "same", activation = "relu");
            layers.MaxPooling2D();*)
            layers.Flatten();
            layers.Dense(128, activation = keras.activations.Relu);
            layers.Dense(num_classes) }))

        model.compile(
            optimizer = keras.optimizers.Adam(),
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits = true),
            metrics = [| "accuracy" |])

        model.summary()
        model

    let private train train_ds val_ds (model : Sequential) =
        model.fit(train_ds, validation_data = val_ds, epochs = epochs)

    let private run () =
        tf.enable_eager_execution()

        let train_ds, val_ds = prepareData()
        let model = buildModel()
        train train_ds val_ds model

        true

    let Example = { Config = ExampleConfig.Create("Image Classification (Keras)", priority = 18)
                    Run = run }

