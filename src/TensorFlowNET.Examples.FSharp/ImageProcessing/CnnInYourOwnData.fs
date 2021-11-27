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
open SharpCV
open System
open System.Diagnostics
open System.IO
open Tensorflow
open Tensorflow.Keras.Utils
open type SharpCV.Binding
open type Tensorflow.Binding

module CnnInYourOwnData =
    let name = "CNN in Your Own Data (Graph)"

    let img_h = 64 // MNIST images are 64x64
    let img_w = 64 // MNIST images are 64x64
    let img_mean = 0
    let img_std = 255
    let n_channels = 1 // Gray Image, one channel

    let display_freq = 2
    let loss_test = 1f

    // Network configuration
    // 1st Convolutional Layer
    let filter_size1 = 5 // Convolution filters are 5 x 5 pixels.
    let num_filters1 = 16 // There are 16 of these filters.
    let stride1 = 1 // The stride of the sliding window

    // 2nd Convolutional Layer
    let filter_size2 = 5 // Convolution filters are 5 x 5 pixels.
    let num_filters2 = 32 // There are 32 of these filters.
    let stride2 = 1 // The stride of the sliding window

    // Fully-connected layer.
    let h1 = 128 // Number of neurons in fully-connected layer.

    // Hyper-parameters 
    let epochs = 5 // accuracy > 98%
    let batch_size = 100
    let mutable learning_rate_base = 0.001f
    let learning_rate_decay = 0.1f
    let learning_rate_step = 2
    let learning_rate_min = 0.000001f

    let SaverBest = true

    let TrainQueueCapa = 3

    let random = Random()

    /// Get Dictionary , Key is Order Number , Value is Label
    let fillDictionaryLabel dirPath =
        let str_dir = Directory.GetDirectories(dirPath, "*", SearchOption.TopDirectoryOnly)
        let dict_Label =
            str_dir
            |> Seq.mapi (fun i dir -> int64 i, dir.Substring(dirPath.Length + 1).Split('\\').[0])
            |> Map.ofSeq
        dict_Label |> Map.iter (fun i label -> print(i.ToString() + " : " + label))
        dict_Label

    /// Get Label Array with Dictionary
    let getLabelArray (dict_index : Map<string, int64>) (files : string[]) =
        files
        |> Array.map (fun x ->
            let labels = x.Split('\\')
            let label = labels.[labels.Length - 2]
            dict_index.[label])

    /// Shuffle Images and Labels Arrays
    let shuffleArrays (images : string[]) (labels : int64[]) =
        let swap (a: _[]) i j =
            let tmp = a.[i]
            a.[i] <- a.[j]
            a.[j] <- tmp

        let len = images.Length
        seq { 0..len - 1 }
        |> Seq.iter (fun i ->
            let j = random.Next(i, len)
            swap images i j
            swap labels i j)
        print("shuffle array list： " + len.ToString())

    let ReadTensorFromImageFile (file_name : string) (graph : Graph) =
        let file_reader = tf.io.read_file(file_name, "file_reader")
        let decodeJpeg = tf.image.decode_jpeg(file_reader, channels = n_channels, name = "DecodeJpeg")
        let cast = tf.cast(decodeJpeg, tf.float32)
        let dims_expander = tf.expand_dims(cast, 0)
        let resize = tf.constant([| img_h; img_w |])
        let bilinear = tf.image.resize_bilinear(dims_expander, resize)
        let sub = tf.subtract(bilinear, [| img_mean |])
        let normalized = tf.divide(sub, [| img_std |])

        use sess = tf.Session(graph)
        sess.run(normalized)

    let LoadImage (a : string[]) (b : NDArray) c =
        use graph = tf.Graph().as_default()
        for i in 0 .. a.Length - 1 do
            let indices : obj[] = [| i |]
            // b.[indices] <- ReadTensorFromImageFile a.[i] graph
            Console.Write(".")
        Console.WriteLine()
        Console.WriteLine("Load Images To NDArray: " + c)

    /// Load Validation and Test data to NDarray, Train data is too large ,we load by training process 
    let LoadImagesToNDArray labelCount arrayFileName_Validation (arrayLabel_Validation : int64[]) arrayFileName_Test (arrayLabel_Test : int64[]) =
        // Load labels
        let y_valid = np.eye(labelCount).[np.array(arrayLabel_Validation)]
        let y_test = np.eye(labelCount).[np.array(arrayLabel_Test)]
        print("Load Labels To NDArray : OK!")

        // Load Images
        let x_valid = np.zeros(arrayLabel_Validation.Length, img_h, img_w, n_channels)
        let x_test = np.zeros(arrayLabel_Test.Length, img_h, img_w, n_channels)
        LoadImage arrayFileName_Validation x_valid "validation"
        LoadImage arrayFileName_Test x_test "test"
        print("Load Images To NDArray : OK!")

        (x_valid, y_valid), (x_test, y_test)

    type InputData = {
        dict_label : Map<int64, string>
        n_classes : int
        x_valid : NDArray
        y_valid : NDArray
        x_test : NDArray
        y_test : NDArray
        ArrayFileName_Train : string[]
        ArrayLabel_Train : int64[]
        ArrayFileName_Test : string[]
        ArrayLabel_Test : int64[]
    }

    let prepareData () =
        let url = "https://github.com/SciSharp/SciSharp-Stack-Examples/raw/master/data/data_CnnInYourOwnData.zip"
        Directory.CreateDirectory(name) |> ignore
        Web.Download(url, name, "data_CnnInYourOwnData.zip") |> ignore
        Compress.UnZip(name + "\\data_CnnInYourOwnData.zip", name)

        let dict_label = fillDictionaryLabel (name + "\\train")
        let n_classes = dict_label.Count
        let dict_index = dict_label |> Map.toSeq |> Seq.map (fun (k, v) -> (v, k)) |> Map.ofSeq

        let arrayFileName_Train = Directory.GetFiles(name + "\\train", "*.*", SearchOption.AllDirectories)
        let arrayLabel_Train = getLabelArray dict_index arrayFileName_Train

        let arrayFileName_Validation = Directory.GetFiles(name + "\\validation", "*.*", SearchOption.AllDirectories)
        let arrayLabel_Validation = getLabelArray dict_index arrayFileName_Validation

        let arrayFileName_Test = Directory.GetFiles(name + "\\test", "*.*", SearchOption.AllDirectories)
        let arrayLabel_Test = getLabelArray dict_index arrayFileName_Test

        // shuffle arrays
        shuffleArrays arrayFileName_Train arrayLabel_Train
        shuffleArrays arrayFileName_Validation arrayLabel_Validation
        shuffleArrays arrayFileName_Test arrayLabel_Test

        let (x_valid, y_valid), (x_test, y_test) = 
            LoadImagesToNDArray n_classes arrayFileName_Validation arrayLabel_Validation arrayFileName_Test arrayLabel_Test

        { dict_label = dict_label
          n_classes = n_classes
          x_valid = x_valid
          y_valid = y_valid
          x_test = x_test
          y_test = y_test
          ArrayFileName_Train = arrayFileName_Train
          ArrayLabel_Train = arrayLabel_Train
          ArrayFileName_Test = arrayFileName_Test
          ArrayLabel_Test = arrayLabel_Test }

    /// Create a weight variable with appropriate initialization
    let private weight_variable name shape =
        let initer = tf.truncated_normal_initializer(stddev = 0.01f)
        tf.compat.v1.get_variable(name,
                                  dtype = tf.float32,
                                  shape = shape,
                                  initializer = initer)

    /// Create a bias variable with appropriate initialization
    let private bias_variable name shape =
        let initial = tf.constant(0f, shape = shape, dtype = tf.float32)
        tf.compat.v1.get_variable(name,
                                  dtype = tf.float32,
                                  initializer = initial)

    /// <summary>
    /// Create a 2D convolution layer
    /// </summary>
    /// <param name="x">input from previous layer</param>
    /// <param name="filter_size">size of each filter</param>
    /// <param name="num_filters">number of filters(or output feature maps)</param>
    /// <param name="stride">filter stride</param>
    /// <param name="name">layer name</param>
    /// <returns>The output array</returns>
    let private conv_layer (x : Tensor) (filter_size : int) (num_filters : int) (stride : int) (name : string) =
        tf_with(tf.variable_scope(name), fun _ ->
            let num_in_channel = x.shape.[x.NDims - 1]
            let shape = TensorShape [| filter_size; filter_size; num_in_channel; num_filters |]
            let W = weight_variable "W" shape
            let b = bias_variable "b" (TensorShape [| num_filters |])
            let layer = tf.nn.conv2d(x, W,
                                     strides = [| 1; stride; stride; 1 |],
                                     padding = "SAME")
            let layer = layer + b.AsTensor()
            tf.nn.relu(layer))

    /// <summary>
    /// Create a max pooling layer
    /// </summary>
    /// <param name="x">input to max-pooling layer</param>
    /// <param name="ksize">size of the max-pooling filter</param>
    /// <param name="stride">stride of the max-pooling filter</param>
    /// <param name="name">layer name</param>
    /// <returns>The output array</returns>
    let private max_pool x ksize stride name =
        tf.nn.max_pool(
            x,
            ksize = [| 1; ksize; ksize; 1 |],
            strides = [| 1; stride; stride; 1 |],
            padding = "SAME",
            name = name)

    /// <summary>
    /// Flattens the output of the convolutional layer to be fed into fully-connected layer
    /// </summary>
    /// <param name="layer">input array</param>
    /// <returns>flattened array</returns>
    let private flatten_layer (layer : Tensor) =
        tf_with(tf.variable_scope("Flatten_layer"), fun _ ->
            let layer_shape = layer.TensorShape
            let num_features = layer_shape.[Slice(1, 4)].size
            let layer_flat = tf.reshape(layer, TensorShape [| -1; num_features |])
            layer_flat
        )

    /// <summary>
    /// Create a fully-connected layer
    /// </summary>
    /// <param name="x">input from previous layer</param>
    /// <param name="num_units">number of hidden units in the fully-connected layer</param>
    /// <param name="name">layer name</param>
    /// <param name="use_relu">boolean to add ReLU non-linearity (or not)</param>
    /// <returns>The output array</returns>
    let private fc_layer (x : Tensor) num_units (name : string) use_relu =
        tf_with(tf.variable_scope(name), fun _ ->
            let in_dim = x.shape.[1]

            let W = weight_variable ("W_" + name) (TensorShape [| in_dim; num_units |])
            let b = bias_variable ("b_" + name) (TensorShape [| num_units |])

            let layer = tf.matmul(x, W.AsTensor()) + b.AsTensor()
            if use_relu then tf.nn.relu(layer) else layer
        )

    type ModelData = {
        x : Tensor
        y : Tensor
        global_steps : IVariableV1
        learning_rate : IVariableV1
        decodeJpeg : Tensor
        normalized : Tensor
        loss : Tensor
        accuracy : Tensor
        cls_prediction : Tensor
        prob : Tensor
    }

    let private buildGraph graph n_classes =

        let mutable tensors = []
        let mutable optimizers = []

        tf_with(tf.name_scope("Input"), fun _ ->
            tensors <-tf.placeholder(tf.float32, shape = TensorShape (-1, img_h, img_w, n_channels), name = "X") :: tensors
            tensors <-tf.placeholder(tf.float32, shape = TensorShape (-1, n_classes), name = "Y") :: tensors
        )

        let x, y =
            match tensors with
            | y :: x :: _ -> x, y
            | _ -> failwith "Unexpected error"

        let conv1 = conv_layer x filter_size1 num_filters1 stride1 "conv1"
        let pool1 = max_pool conv1 2 2 "pool1"
        let conv2 = conv_layer pool1 filter_size2 num_filters2 stride2 "conv2"
        let pool2 = max_pool conv2 2 2 "pool2"
        let layer_flat = flatten_layer pool2
        let fc1 = fc_layer layer_flat h1 "FC1" true
        let output_logits = fc_layer fc1 n_classes "OUT" false

        // Some important parameter saved with graph , easy to load later
        //let img_h_t = tf.constant(img_h, name = "img_h")
        //let img_w_t = tf.constant(img_w, name = "img_w")
        //let img_mean_t = tf.constant(img_mean, name = "img_mean")
        //let img_std_t = tf.constant(img_std, name = "img_std")
        //let channels_t = tf.constant(n_channels, name = "img_channels")

        // learning rate decay
        let global_steps = tf.Variable(0, trainable = false)
        let learning_rate = tf.Variable(learning_rate_base)

        // create train images graph
        tf_with(tf.variable_scope("LoadImage"), fun _ ->
            tensors <- tf.placeholder(tf.byte8, name = "DecodeJpeg") :: tensors
            let decodeJpeg = List.head tensors
            let cast = tf.cast(decodeJpeg, tf.float32)
            let dims_expander = tf.expand_dims(cast, 0)
            let resize = tf.constant([| img_h; img_w |])
            let bilinear = tf.image.resize_bilinear(dims_expander, resize)
            let sub = tf.subtract(bilinear, [| img_mean |])
            tensors <- tf.divide(sub, [| img_std |], name = "normalized") :: tensors)

        tf_with(tf.variable_scope("Train"), fun _ ->
            tf_with(tf.variable_scope("Loss"), fun _ ->
                tensors <- tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = output_logits), name = "loss") :: tensors
            )

            let loss = List.head tensors

            tf_with(tf.variable_scope("Optimizer"), fun _ ->
                optimizers <- tf.train.AdamOptimizer(learning_rate = learning_rate, name = "Adam-op").minimize(loss, global_step = global_steps) :: optimizers
            )

            tf_with(tf.variable_scope("Accuracy"), fun _ ->
                let correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name = "correct_pred")
                tensors <- tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = "accuracy") :: tensors
            )

            tf_with(tf.variable_scope("Prediction"), fun _ ->
                tensors <- tf.argmax(output_logits, axis = 1, name = "predictions") :: tensors
                tensors <- tf.nn.softmax(output_logits, axis = 1, name = "prob") :: tensors
            ))

        match tensors, optimizers with
        | prob :: cls_prediction :: accuracy :: loss :: normalized :: decodeJpeg :: y :: x :: _, optimizer :: _ ->
            { x = x
              y = y
              global_steps = global_steps
              learning_rate = learning_rate
              decodeJpeg = decodeJpeg
              normalized = normalized
              loss = loss
              accuracy = accuracy
              cls_prediction = cls_prediction
              prob = prob }, optimizer
        | _ -> failwith "Unexpexted error"

    let private write_Dictionary path (mydic : Map<int64, string>) =
        let lines =
            mydic
            |> Map.toSeq
            |> Seq.map (fun (k, v) -> sprintf "%i,%s" k v)
        File.WriteAllLines(path, lines)
        print("write_Dictionary")

    let private train (inputData : InputData) (modelData : ModelData) (optimizer : Operation) (sess : Session) =
        // Number of training iterations in each epoch
        let num_tr_iter = inputData.ArrayLabel_Train.Length / batch_size

        let init = tf.global_variables_initializer()
        sess.run(init)

        let saver = tf.train.Saver(tf.global_variables(), max_to_keep = 10)

        let path_model = name + "\\MODEL"
        Directory.CreateDirectory(path_model) |> ignore

        let mutable max_accuracy = 0f

        let sw = Stopwatch.StartNew()
        for epoch in range(epochs) do
            print($"Training epoch: {epoch + 1}")
            // Randomly shuffle the training data at the beginning of each epoch 
            shuffleArrays inputData.ArrayFileName_Train inputData.ArrayLabel_Train
            let y_train = np.eye(inputData.n_classes).[new NDArray(inputData.ArrayLabel_Train)]

            // decay learning rate
            if learning_rate_step <> 0 then
                if epoch <> 0 && epoch % learning_rate_step = 0 then
                    learning_rate_base <- learning_rate_base * learning_rate_decay
                    if learning_rate_base <= learning_rate_min then learning_rate_base <- learning_rate_min
                    sess.run(tf.assign(modelData.learning_rate, learning_rate_base)) |> ignore

            let GetNextBatch (x : string[]) (y : NDArray) start (``end`` : int) =
                let x_batch = np.zeros(``end`` - start, img_h, img_w, n_channels)
                for i in start .. ``end`` - 1 do
                    let n : obj[] = [| i - start |]
                    let img4 = cv2.imread(x.[i], IMREAD_COLOR.IMREAD_GRAYSCALE).asNDArray
                    let img4 = img4.reshape(img4.shape.[0], img4.shape.[1], 1)
                    x_batch.[n] <- sess.run(modelData.normalized, feedItems [| modelData.decodeJpeg, img4 |])
                let slice = Slice(start, ``end``)
                let y_batch = y.[slice]
                (x_batch, y_batch)

            // Load local images asynchronously in parallel to improve train efficiency
            let batches =
                range(num_tr_iter)
                |> Seq.map (fun iteration -> async {
                    let start = iteration * batch_size
                    let ``end`` = (iteration + 1) * batch_size
                    let (x_batch, y_batch) = GetNextBatch inputData.ArrayFileName_Train y_train start ``end``
                    return (x_batch, y_batch, iteration)
                })
                |> Async.Parallel
                |> Async.RunSynchronously

            for c_x, c_y, iter in batches do
                sess.run(optimizer, feedItems [| modelData.x, c_x; modelData.y, c_y |]) |> ignore

                if iter % display_freq = 0 then
                    // Calculate and display the batch loss and accuracy
                    let result = sess.run([| modelData.loss; modelData.accuracy |], feedItems [| (modelData.x, c_x); (modelData.y, c_y) |])
                    let loss_val, accuracy_val = float32 result.[0], float32 result.[1]
                    print($"""CNN：iter {iter.ToString("000")}: Loss={loss_val.ToString("0.0000")}, Training Accuracy={accuracy_val.ToString("P")} {sw.ElapsedMilliseconds}ms""")
                    sw.Restart()

            // Run validation after every epoch
            let struct (loss_val, accuracy_val) = sess.run(fetches (modelData.loss, modelData.accuracy), feedItems [| modelData.x, inputData.x_valid; modelData.y, inputData.y_valid |])
            print("CNN：---------------------------------------------------------")
            print($"""CNN：global steps: {int <| sess.run(modelData.global_steps).[0] }, learning rate: {float32 <| sess.run(modelData.learning_rate).[0]}, validation loss: {(float32 loss_val).ToString("0.0000")}, validation accuracy: {(float32 accuracy_val).ToString("P")}""")
            print("CNN：---------------------------------------------------------")

            let accuracy = float32 accuracy_val
            let loss = float32 loss_val
            if SaverBest then
                if accuracy > max_accuracy then
                    max_accuracy <- accuracy
                    saver.save(sess, path_model + "\\CNN_Best") |> ignore
                    print("CKPT Model is saved.")
            else
                saver.save(sess, sprintf "%s\\CNN_Epoch_%i_Loss_%f_Acc_%f" path_model epoch loss accuracy) |> ignore
                print("CKPT Model is saved.")
        write_Dictionary (path_model + "\\dic.txt") inputData.dict_label

    let private test (inputData : InputData) (modelData : ModelData) (sess : Session) =
        let struct (loss_test, accuracy_test) = sess.run(fetches (modelData.loss, modelData.accuracy), feedItems [| modelData.x, inputData.x_test; modelData.y, inputData.y_test |])
        let accuracy_test = float32 accuracy_test
        print("CNN：---------------------------------------------------------")
        print($"""CNN：Test loss: {(float32 loss_test).ToString("0.0000")}, test accuracy: {accuracy_test.ToString("P")}""")
        print("CNN：---------------------------------------------------------")

        let struct (test_cls, test_data) = sess.run(fetches (modelData.cls_prediction, modelData.prob), feedItems [| modelData.x, inputData.x_test |])
        (test_cls, test_data, accuracy_test)

    let private testDataOutput (inputData : InputData) (test_cls : NDArray) (test_data : NDArray) =
        for i in 0 .. inputData.ArrayLabel_Test.Length - 1 do
            let real = inputData.ArrayLabel_Test.[i]
            let predict = test_cls.[i]
            let predict64 = int64 predict
            let probability = test_data.[i, predict]
            let result = if real = predict64 then "OK" else "NG"
            let fileName = inputData.ArrayFileName_Test.[i]
            let real_str = inputData.dict_label.[real]
            let predict_str = inputData.dict_label.[predict64]
            print((i + 1).ToString() + "|" + "result:" + result + "|" + "real_str:" + real_str + "|"
                + "predict_str:" + predict_str + "|" + "probability:" + probability.GetSingle().ToString() + "|"
                + "fileName:" + fileName)

    let private run () =
        tf.compat.v1.disable_eager_execution()

        let inputData = prepareData()

        use g = new Graph()
        let graph = g.as_default()

        let modelData, optimizer = buildGraph graph inputData.n_classes

        let sess = tf.Session()

        train inputData modelData optimizer sess
        let test_cls, test_data, accuracy_test = test inputData modelData sess

        testDataOutput inputData test_cls test_data

        accuracy_test > 0.98f

    let Example = { Config = ExampleConfig.Create(name, priority = 19)
                    Run = run }

