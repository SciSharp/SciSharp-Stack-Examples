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
open type Tensorflow.Binding

module DigitRecognitionCNN =

    let img_h, img_w = 28, 28; // MNIST images are 28x28
    let n_classes = 10 // Number of classes, one class per digit
    let n_channels = 1

    // Hyper-parameters
    let epochs = 5 // accuracy > 98%
    let batch_size = 100
    let learning_rate = 0.001f
    //Datasets<MnistDataSet> mnist;

    // Network configuration
    // 1st Convolutional Layer
    let filter_size1 = 5  // Convolution filters are 5 x 5 pixels.
    let num_filters1 = 16 // There are 16 of these filters.
    let stride1 = 1 // The stride of the sliding window

    // 2nd Convolutional Layer
    let filter_size2 = 5 // Convolution filters are 5 x 5 pixels.
    let num_filters2 = 32 // There are 32 of these filters.
    let stride2 = 1  // The stride of the sliding window

    // Fully-connected layer.
    let h1 = 128 // Number of neurons in fully-connected layer.

    //Tensor x, y;
    //Tensor loss, accuracy, cls_prediction;
    //Operation optimizer;

    let display_freq = 100
    //let accuracy_test = 0f
    let loss_test = 1f

    let name = "MNIST CNN (Graph)"

    /// Reformats the data to the format acceptable for convolutional layers
    let Reformat (x : NDArray) y =
        //let num_class = len(np.unique(&(np.argmax(y, 1))))
        let dataset = x.reshape(x.shape.[0], img_h, img_w, 1).astype(np.float32)
        //y[0] = np.arange(num_class) == y[0]
        //var labels = (np.arange(num_class) == y.reshape(y.shape[0], 1, y.shape[1])).astype(np.float32)
        dataset, y

    let private prepareData () =
        Directory.CreateDirectory(name) |> ignore

        let mnist = MnistModelLoader.LoadAsync(".resources/mnist", oneHot = true, showProgressInConsole = true).Result
        let x_train, y_train = Reformat mnist.Train.Data mnist.Train.Labels
        let x_valid, y_valid = Reformat mnist.Validation.Data mnist.Validation.Labels
        let x_test, y_test = Reformat mnist.Test.Data mnist.Test.Labels

        print("Size of:")
        print($"- Training-set:\t\t{len(mnist.Train.Data)}")
        print($"- Validation-set:\t{len(mnist.Validation.Data)}")

        mnist, (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

    /// Create a weight variable with appropriate initialization
    let private weight_variable name shape =
        let initer = tf.truncated_normal_initializer(stddev = 0.01f)
        tf.compat.v1.get_variable(name, dtype = tf.float32, shape = shape, initializer = initer)

    /// <summary>
    /// Create a bias variable with appropriate initialization
    /// </summary>
    /// <param name="name"></param>
    /// <param name="shape"></param>
    /// <returns></returns>
    let private bias_variable name shape =
        let initial = tf.constant(0f, shape = shape, dtype = tf.float32)
        tf.compat.v1.get_variable(name, dtype = tf.float32, initializer = initial)

    /// <summary>
    /// Create a 2D convolution layer
    /// </summary>
    /// <param name="x">input from previous layer</param>
    /// <param name="filter_size">size of each filter</param>
    /// <param name="num_filters">number of filters(or output feature maps)</param>
    /// <param name="stride">filter stride</param>
    /// <param name="name">layer name</param>
    /// <returns>The output array</returns>
    let private conv_layer (x : Tensor) filter_size num_filters stride (name : string) =
        tf_with(tf.variable_scope(name), fun _ ->
            let num_in_channel = x.shape.[x.NDims - 1]
            let shape = TensorShape [| filter_size; filter_size; num_in_channel; num_filters |]
            let W = weight_variable "W" shape
            // tf.summary.histogram("weight", W)
            let b = bias_variable "b" (TensorShape [| num_filters |])
            // tf.summary.histogram("bias", b)
            let layer = tf.nn.conv2d(x, W,
                                     strides = [| 1; stride; stride; 1 |],
                                     padding = "SAME")
            let layer = layer + b.AsTensor()
            tf.nn.relu(layer)
        )

    /// <summary>
    /// Create a max pooling layer
    /// </summary>
    /// <param name="x">input to max-pooling layer</param>
    /// <param name="ksize">size of the max-pooling filter</param>
    /// <param name="stride">stride of the max-pooling filter</param>
    /// <param name="name">layer name</param>
    /// <returns>The output array</returns>
    let private max_pool x ksize stride name =
        tf.nn.max_pool(x, ksize = [| 1; ksize; ksize; 1 |], strides = [| 1; stride; stride; 1 |], padding = "SAME", name = name)

    /// <summary>
    /// Flattens the output of the convolutional layer to be fed into fully-connected layer
    /// </summary>
    /// <param name="layer">input array</param>
    /// <returns>flattened array</returns>
    let private flatten_layer (layer : Tensor) =
        tf_with(tf.variable_scope("Flatten_layer"), fun _ ->
            let layer_shape = layer.TensorShape
            let num_features = layer_shape.[new Slice(1, 4)].size
            tf.reshape(layer, TensorShape [| -1; num_features |]))

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
            if use_relu then
                tf.nn.relu(layer)
            else
                layer;
        )

    let private buildGraph graph =

        let mutable tensors = []
        let mutable operations = []

        tf_with(tf.name_scope("Input"), fun _ ->
            // Placeholders for inputs (x) and outputs(y)
            tensors <- tf.placeholder(tf.float32, shape = TensorShape (-1, img_h, img_w, n_channels), name = "X") :: tensors
            tensors <- tf.placeholder(tf.float32, shape = TensorShape (-1, n_classes), name = "Y") :: tensors
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

        tf_with(tf.variable_scope("Train"), fun _ ->

            tf_with(tf.variable_scope("Loss"), fun _ ->
                tensors <- tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = output_logits), name = "loss") :: tensors
            )

            let loss = List.head tensors

            tf_with(tf.variable_scope("Optimizer"), fun _ ->
                operations <- tf.train.AdamOptimizer(learning_rate = learning_rate, name = "Adam-op").minimize(loss) :: operations
            )

            tf_with(tf.variable_scope("Accuracy"), fun _ ->
                let correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name = "correct_pred")
                tensors <- tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = "accuracy") :: tensors
            )

            tf_with(tf.variable_scope("Prediction"), fun _ ->
                tensors <- tf.argmax(output_logits, axis = 1, name = "predictions") :: tensors
            )
        )

        match tensors, operations with
        | cls_prediction :: accuracy :: loss :: y :: x :: _, optimizer :: _ ->
            (x, y), (loss, accuracy, cls_prediction), optimizer
        | _ -> failwith "Unexpected error"

    let SaveCheckpoint sess =
        let saver = tf.train.Saver()
        saver.save(sess, Path.Combine(name, "mnist_cnn.ckpt")) |> ignore

    let private train (mnist : Datasets<MnistDataSet>) (x_train : NDArray, y_train : NDArray) (x_valid, y_valid) =
        use g = new Graph()
        let graph = g.as_default()

        let (x, y), (loss, accuracy, cls_prediction), optimizer = buildGraph graph

        use sess = tf.Session(graph)

        // Number of training iterations in each epoch
        let num_tr_iter = y_train.shape.[0] / batch_size

        let init = tf.global_variables_initializer()
        sess.run(init)

        let sw = Stopwatch.StartNew()
        for epoch in range(epochs) do

            print($"Training epoch: {epoch + 1}")
            // Randomly shuffle the training data at the beginning of each epoch 
            let struct (x_train, y_train) = mnist.Randomize(x_train, y_train)

            for iteration in range(num_tr_iter) do

                let start = iteration * batch_size
                let ``end`` = (iteration + 1) * batch_size
                let struct (x_batch, y_batch) = mnist.GetNextBatch(x_train, y_train, start, ``end``)

                // Run optimization op (backprop)
                sess.run(optimizer, feedItems [| x, x_batch; y, y_batch |])

                if iteration % display_freq = 0 then

                    // Calculate and display the batch loss and accuracy
                    let struct (loss_val, accuracy_val) = sess.run(fetches (loss, accuracy), feedItems [| x, x_batch; y, y_batch |])
                    let loss_val, accuracy_val = float32 loss_val, float32 accuracy_val
                    print($"""iter {iteration.ToString("000")}: Loss={loss_val.ToString("0.0000")}, Training Accuracy={accuracy_val.ToString("P")} {sw.ElapsedMilliseconds}ms""")
                    sw.Restart()

            // Run validation after every epoch
            let struct (loss_val, accuracy_val) = sess.run(fetches (loss, accuracy), feedItems [| x, x_valid; y, y_valid |])
            let loss_val, accuracy_val = float32 loss_val, float32 accuracy_val
            print("---------------------------------------------------------")
            print($"""Epoch: {epoch + 1}, validation loss: {loss_val.ToString("0.0000")}, validation accuracy: {accuracy_val.ToString("P")}""")
            print("---------------------------------------------------------")

        SaveCheckpoint(sess)

    let private test (x_test, y_test) =
        use graph = tf.Graph().as_default()
        use sess = tf.Session(graph)

        let saver = tf.train.import_meta_graph(Path.Combine(name, "mnist_cnn.ckpt.meta"))
        // Restore variables from checkpoint
        saver.restore(sess, tf.train.latest_checkpoint(name))

        let loss = graph.get_tensor_by_name("Train/Loss/loss:0")
        let accuracy = graph.get_tensor_by_name("Train/Accuracy/accuracy:0")
        let x = graph.get_tensor_by_name("Input/X:0")
        let y = graph.get_tensor_by_name("Input/Y:0")

        //let init = tf.global_variables_initializer()
        //sess.run(init)

        let struct (loss_test, accuracy_test) = sess.run(fetches (loss, accuracy), feedItems [| x, x_test; y, y_test |])
        let loss_test, accuracy_test = float32 loss_test, float32 accuracy_test
        print("---------------------------------------------------------")
        print($"""Test loss: {loss_test.ToString("0.0000")}, test accuracy: {accuracy_test.ToString("P")}""")
        print("---------------------------------------------------------")
        
        accuracy_test

    let private run () =
        tf.compat.v1.disable_eager_execution()

        let mnist, (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = prepareData()

        train mnist (x_train, y_train) (x_valid, y_valid)
        let accuracy_test = test (x_test, y_test)

        accuracy_test > 0.98f

    let Example = { Config = ExampleConfig.Create(name, priority = 15)
                    Run = run }

