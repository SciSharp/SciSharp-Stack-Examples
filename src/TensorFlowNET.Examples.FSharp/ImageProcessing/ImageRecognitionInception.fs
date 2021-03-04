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
open Tensorflow.Keras.Utils
open type Tensorflow.Binding

/// <summary>
/// Inception v3 is a widely-used image recognition model 
/// that has been shown to attain greater than 78.1% accuracy on the ImageNet dataset. 
/// The model is the culmination of many ideas developed by multiple researchers over the years.
/// </summary>
module ImageRecognitionInception =
    let dir = "ImageRecognitionInception"
    let pbFile = "tensorflow_inception_graph.pb"
    let labelFile = "imagenet_comp_graph_label_strings.txt"
 
    let ReadTensorFromImageFile (file_name : string) =
        let input_height = 224
        let input_width = 224
        let input_mean = 117
        let input_std = 1

        let graph = tf.Graph().as_default()

        let file_reader = tf.io.read_file(file_name, "file_reader")
        let decodeJpeg = tf.image.decode_jpeg(file_reader, channels = 3, name = "DecodeJpeg")
        let cast = tf.cast(decodeJpeg, tf.float32)
        let dims_expander = tf.expand_dims(cast, 0)
        let resize = tf.constant([| input_height; input_width |])
        let bilinear = tf.image.resize_bilinear(dims_expander, resize)
        let sub = tf.subtract(bilinear, [| input_mean |])
        let normalized = tf.divide(sub, [| input_std |])

        use sess = tf.Session(graph)
        sess.run(normalized)

    let prepareData () =
        Directory.CreateDirectory(dir) |> ignore

        // get model file
        let url = "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"
        Web.Download(url, dir, "inception5h.zip") |> ignore

        Compress.UnZip(Path.Join(dir, "inception5h.zip"), dir)

        // download sample picture
        Directory.CreateDirectory(Path.Join(dir, "img")) |> ignore
        let url = $"https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/label_image/data/grace_hopper.jpg"
        Web.Download(url, Path.Join(dir, "img"), "grace_hopper.jpg") |> ignore

        let url = $"https://raw.githubusercontent.com/SciSharp/TensorFlow.NET/master/data/shasta-daisy.jpg"
        Web.Download(url, Path.Join(dir, "img"), "shasta-daisy.jpg") |> ignore

        // load image file
        Directory.GetFiles(Path.Join(dir, "img"))
        |> Array.map (fun f -> ReadTensorFromImageFile(f))

    let private run () =
        tf.compat.v1.disable_eager_execution()

        let file_ndarrays = prepareData()

        use graph = new Graph()
        //import GraphDef from pb file
        graph.Import(Path.Join(dir, pbFile)) |> ignore

        let input_name = "input"
        let output_name = "output"

        let input_operation = graph.OperationByName(input_name)
        let output_operation = graph.OperationByName(output_name)

        let labels = File.ReadAllLines(Path.Join(dir, labelFile))
        let sw = new Stopwatch()

        use sess = tf.Session(graph)
        let result_labels =
            file_ndarrays
            |> Array.map (fun nd ->
                sw.Restart()

                let results = sess.run(output_operation.outputs.[0], feedItems [| input_operation.outputs.[0], nd |])
                let results = np.squeeze(results)
                let idx = np.argmax(results)

                Console.WriteLine($"{labels.[idx]} {results.[idx]} in {sw.ElapsedMilliseconds}ms", Color.Tan)
                labels.[idx])
            |> Set.ofArray

        result_labels.Contains("military uniform")

    let Example = { Config = ExampleConfig.Create("Image Recognition Inception")
                    Run = run }
