/*****************************************************************************
  Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
******************************************************************************/


using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using SharpCV;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using Tensorflow;
using static Tensorflow.Binding;
using static SharpCV.Binding;
using System.Collections.Concurrent;
using System.Threading.Tasks;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Convolutional Neural Network classifier for Local Classify Images.
    /// CNN architecture with two convolutional layers, followed by two fully-connected layers at the end.
    /// Use Stochastic Gradient Descent (SGD) optimizer. 
    /// Learning Rate reduces per 10 epochs.
    /// TODO: Thread Queue will be added to improve images loading efficiency later.
    /// https://www.easy-tensorflow.com/tf-tutorials/convolutional-neural-nets-cnns/cnn1
    /// </summary>
    class CnnInYourOwnData : SciSharpExample, IExample
    {
        string logs_path = "logs";

        string[] ArrayFileName_Train, ArrayFileName_Validation, ArrayFileName_Test;
        Int64[] ArrayLabel_Train, ArrayLabel_Validation, ArrayLabel_Test;
        Dictionary<Int64, string> Dict_Label;
        NDArray x_train, y_train;
        NDArray x_valid, y_valid;
        NDArray x_test, y_test;
        int img_h = 64;// MNIST images are 64x64
        int img_w = 64;// MNIST images are 64x64
        int img_mean = 0;
        int img_std = 255;
        int n_channels = 1;//Gray Image ,one channel
        int n_classes;// Number of classes

        Tensor x, y; // Placeholders for inputs (x) and outputs(y)
        Tensor loss, accuracy, cls_prediction, prob;
        Tensor optimizer;
        Tensor normalized;
        Tensor decodeJpeg;

        int display_freq = 2;
        float accuracy_test = 0f;
        float loss_test = 1f;

        // Network configuration
        // 1st Convolutional Layer
        int filter_size1 = 5;  // Convolution filters are 5 x 5 pixels.
        int num_filters1 = 16; //  There are 16 of these filters.
        int stride1 = 1;  // The stride of the sliding window

        // 2nd Convolutional Layer
        int filter_size2 = 5; // Convolution filters are 5 x 5 pixels.
        int num_filters2 = 32;// There are 32 of these filters.
        int stride2 = 1;  // The stride of the sliding window

        // Fully-connected layer.
        int h1 = 128; // Number of neurons in fully-connected layer.

        // Hyper-parameters 
        int epochs = 5; // accuracy > 98%
        int batch_size = 100;
        float learning_rate_base = 0.001f;
        float learning_rate_decay = 0.1f;
        uint learning_rate_step = 2;
        float learning_rate_min = 0.000001f;

        NDArray Test_Cls, Test_Data;

        RefVariable gloabl_steps;
        RefVariable learning_rate;

        bool SaverBest = true;
        double max_accuracy = 0;

        string path_model;
        int TrainQueueCapa = 3;
        Session sess;

        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "CnnInYourOwnData",
                Enabled = true,
                IsImportingGraph = false,
                Priority = 12
            };

        public bool Run()
        {
            PrepareData();
            BuildGraph();

            sess = tf.Session();

            Train();
            Test();

            TestDataOutput();

            return accuracy_test > 0.98;

        }

        #region PrepareData
        public override void PrepareData()
        {
            string url = "https://github.com/SciSharp/SciSharp-Stack-Examples/raw/master/data/data_CnnInYourOwnData.zip";
            Directory.CreateDirectory(Config.Name);
            Utility.Web.Download(url, Config.Name, "data_CnnInYourOwnData.zip");
            Utility.Compress.UnZip(Config.Name + "\\data_CnnInYourOwnData.zip", Config.Name);

            FillDictionaryLabel(Config.Name + "\\train");

            ArrayFileName_Train = Directory.GetFiles(Config.Name + "\\train", "*.*", SearchOption.AllDirectories);
            ArrayLabel_Train = GetLabelArray(ArrayFileName_Train);

            ArrayFileName_Validation = Directory.GetFiles(Config.Name + "\\validation", "*.*", SearchOption.AllDirectories);
            ArrayLabel_Validation = GetLabelArray(ArrayFileName_Validation);

            ArrayFileName_Test = Directory.GetFiles(Config.Name + "\\test", "*.*", SearchOption.AllDirectories);
            ArrayLabel_Test = GetLabelArray(ArrayFileName_Test);

            //shuffle array
            (ArrayFileName_Train, ArrayLabel_Train) = ShuffleArray(ArrayLabel_Train.Length, ArrayFileName_Train, ArrayLabel_Train);
            (ArrayFileName_Validation, ArrayLabel_Validation) = ShuffleArray(ArrayLabel_Validation.Length, ArrayFileName_Validation, ArrayLabel_Validation);
            (ArrayFileName_Test, ArrayLabel_Test) = ShuffleArray(ArrayLabel_Test.Length, ArrayFileName_Test, ArrayLabel_Test);

            LoadImagesToNDArray();
        }
        /// <summary>
        /// Load Validation and Test data to NDarray, Train data is too large ,we load by training process 
        /// </summary>
        private void LoadImagesToNDArray()
        {
            //Load labels
            y_valid = np.eye(Dict_Label.Count)[new NDArray(ArrayLabel_Validation)];
            y_test = np.eye(Dict_Label.Count)[new NDArray(ArrayLabel_Test)];
            print("Load Labels To NDArray : OK!");

            //Load Images
            x_valid = np.zeros(ArrayFileName_Validation.Length, img_h, img_w, n_channels);
            x_test = np.zeros(ArrayFileName_Test.Length, img_h, img_w, n_channels);
            LoadImage(ArrayFileName_Validation, x_valid, "validation");
            LoadImage(ArrayFileName_Test, x_test, "test");
            print("Load Images To NDArray : OK!");
        }
        private void LoadImage(string[] a, NDArray b, string c)
        {
            using (var graph = tf.Graph().as_default())
            {
                for (int i = 0; i < a.Length; i++)
                {
                    b[i] = ReadTensorFromImageFile(a[i], graph);
                    Console.Write(".");
                }
            }

            Console.WriteLine();
            Console.WriteLine("Load Images To NDArray: " + c);
        }

        private NDArray ReadTensorFromImageFile(string file_name, Graph graph)
        {
            var file_reader = tf.read_file(file_name, "file_reader");
            var decodeJpeg = tf.image.decode_jpeg(file_reader, channels: n_channels, name: "DecodeJpeg");
            var cast = tf.cast(decodeJpeg, tf.float32);
            var dims_expander = tf.expand_dims(cast, 0);
            var resize = tf.constant(new int[] { img_h, img_w });
            var bilinear = tf.image.resize_bilinear(dims_expander, resize);
            var sub = tf.subtract(bilinear, new float[] { img_mean });
            var normalized = tf.divide(sub, new float[] { img_std });

            using (var sess = tf.Session(graph))
            {
                return sess.run(normalized);
            }
        }

        /// <summary>
        /// Shuffle Images and Labels Array
        /// </summary>
        /// <param name="count"></param>
        /// <param name="images"></param>
        /// <param name="labels"></param>
        /// <returns></returns>
        public (string[], Int64[]) ShuffleArray(int count, string[] images, Int64[] labels)
        {
            ArrayList mylist = new ArrayList();
            string[] new_images = new string[count];
            Int64[] new_labels = new Int64[count];
            Random r = new Random();
            for (int i = 0; i < count; i++)
            {
                mylist.Add(i);
            }

            for (int i = 0; i < count; i++)
            {
                int rand = r.Next(mylist.Count);
                new_images[i] = images[(int)(mylist[rand])];
                new_labels[i] = labels[(int)(mylist[rand])];
                mylist.RemoveAt(rand);
            }
            print("shuffle array list： " + count.ToString());
            return (new_images, new_labels);
        }

        /// <summary>
        /// Get Label Array with Dictionary
        /// </summary>
        /// <param name="FilesArray"></param>
        /// <returns></returns>
        private Int64[] GetLabelArray(string[] FilesArray)
        {
            Int64[] ArrayLabel = new Int64[FilesArray.Length];
            for (int i = 0; i < ArrayLabel.Length; i++)
            {
                string[] labels = FilesArray[i].Split('\\');
                string label = labels[labels.Length - 2];
                ArrayLabel[i] = Dict_Label.Single(k => k.Value == label).Key;
            }
            return ArrayLabel;
        }

        /// <summary>
        /// Get Dictionary , Key is Order Number , Value is Label
        /// </summary>
        /// <param name="DirPath"></param>
        private void FillDictionaryLabel(string DirPath)
        {
            string[] str_dir = Directory.GetDirectories(DirPath, "*", SearchOption.TopDirectoryOnly);
            int str_dir_num = str_dir.Length;
            if (str_dir_num > 0)
            {
                Dict_Label = new Dictionary<Int64, string>();
                for (int i = 0; i < str_dir_num; i++)
                {
                    string label = (str_dir[i].Replace(DirPath + "\\", "")).Split('\\').First();
                    Dict_Label.Add(i, label);
                    print(i.ToString() + " : " + label);
                }
                n_classes = Dict_Label.Count;
            }
        }
        #endregion

        #region BuildGraph
        public override Graph BuildGraph()
        {
            var graph = new Graph().as_default();

            tf_with(tf.name_scope("Input"), delegate
            {
                x = tf.placeholder(tf.float32, shape: (-1, img_h, img_w, n_channels), name: "X");
                y = tf.placeholder(tf.float32, shape: (-1, n_classes), name: "Y");
            });

            var conv1 = conv_layer(x, filter_size1, num_filters1, stride1, name: "conv1");
            var pool1 = max_pool(conv1, ksize: 2, stride: 2, name: "pool1");
            var conv2 = conv_layer(pool1, filter_size2, num_filters2, stride2, name: "conv2");
            var pool2 = max_pool(conv2, ksize: 2, stride: 2, name: "pool2");
            var layer_flat = flatten_layer(pool2);
            var fc1 = fc_layer(layer_flat, h1, "FC1", use_relu: true);
            var output_logits = fc_layer(fc1, n_classes, "OUT", use_relu: false);

            //Some important parameter saved with graph , easy to load later
            var img_h_t = tf.constant(img_h, name: "img_h");
            var img_w_t = tf.constant(img_w, name: "img_w");
            var img_mean_t = tf.constant(img_mean, name: "img_mean");
            var img_std_t = tf.constant(img_std, name: "img_std");
            var channels_t = tf.constant(n_channels, name: "img_channels");

            //learning rate decay
            gloabl_steps = tf.Variable(0, trainable: false);
            learning_rate = tf.Variable(learning_rate_base);

            //create train images graph
            tf_with(tf.variable_scope("LoadImage"), delegate
            {
                decodeJpeg = tf.placeholder(tf.@byte, name: "DecodeJpeg");
                var cast = tf.cast(decodeJpeg, tf.float32);
                var dims_expander = tf.expand_dims(cast, 0);
                var resize = tf.constant(new int[] { img_h, img_w });
                var bilinear = tf.image.resize_bilinear(dims_expander, resize);
                var sub = tf.subtract(bilinear, new float[] { img_mean });
                normalized = tf.divide(sub, new float[] { img_std }, name: "normalized");
            });

            tf_with(tf.variable_scope("Train"), delegate
            {
                tf_with(tf.variable_scope("Loss"), delegate
                {
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels: y, logits: output_logits), name: "loss");
                });

                tf_with(tf.variable_scope("Optimizer"), delegate
                {
                    optimizer = tf.train.AdamOptimizer(learning_rate: learning_rate, name: "Adam-op").minimize(loss, global_step: gloabl_steps);
                });

                tf_with(tf.variable_scope("Accuracy"), delegate
                {
                    var correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name: "correct_pred");
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name: "accuracy");
                });

                tf_with(tf.variable_scope("Prediction"), delegate
                {
                    cls_prediction = tf.argmax(output_logits, axis: 1, name: "predictions");
                    prob = tf.nn.softmax(output_logits, axis: 1, name: "prob");
                });
            });
            return graph;
        }

        /// <summary>
        /// Create a 2D convolution layer
        /// </summary>
        /// <param name="x">input from previous layer</param>
        /// <param name="filter_size">size of each filter</param>
        /// <param name="num_filters">number of filters(or output feature maps)</param>
        /// <param name="stride">filter stride</param>
        /// <param name="name">layer name</param>
        /// <returns>The output array</returns>
        private Tensor conv_layer(Tensor x, int filter_size, int num_filters, int stride, string name)
        {
            return tf_with(tf.variable_scope(name), delegate
            {

                var num_in_channel = x.shape[x.NDims - 1];
                var shape = new[] { filter_size, filter_size, num_in_channel, num_filters };
                var W = weight_variable("W", shape);
                // var tf.summary.histogram("weight", W);
                var b = bias_variable("b", new[] { num_filters });
                // tf.summary.histogram("bias", b);
                var layer = tf.nn.conv2d(x, W,
                                     strides: new[] { 1, stride, stride, 1 },
                                     padding: "SAME");
                layer += b;
                return tf.nn.relu(layer);
            });
        }

        /// <summary>
        /// Create a max pooling layer
        /// </summary>
        /// <param name="x">input to max-pooling layer</param>
        /// <param name="ksize">size of the max-pooling filter</param>
        /// <param name="stride">stride of the max-pooling filter</param>
        /// <param name="name">layer name</param>
        /// <returns>The output array</returns>
        private Tensor max_pool(Tensor x, int ksize, int stride, string name)
        {
            return tf.nn.max_pool(x,
                ksize: new[] { 1, ksize, ksize, 1 },
                strides: new[] { 1, stride, stride, 1 },
                padding: "SAME",
                name: name);
        }

        /// <summary>
        /// Flattens the output of the convolutional layer to be fed into fully-connected layer
        /// </summary>
        /// <param name="layer">input array</param>
        /// <returns>flattened array</returns>
        private Tensor flatten_layer(Tensor layer)
        {
            return tf_with(tf.variable_scope("Flatten_layer"), delegate
            {
                var layer_shape = layer.TensorShape;
                var num_features = layer_shape[new Slice(1, 4)].size;
                var layer_flat = tf.reshape(layer, new[] { -1, num_features });

                return layer_flat;
            });
        }

        /// <summary>
        /// Create a weight variable with appropriate initialization
        /// </summary>
        /// <param name="name"></param>
        /// <param name="shape"></param>
        /// <returns></returns>
        private RefVariable weight_variable(string name, int[] shape)
        {
            var initer = tf.truncated_normal_initializer(stddev: 0.01f);
            return tf.get_variable(name,
                                   dtype: tf.float32,
                                   shape: shape,
                                   initializer: initer);
        }

        /// <summary>
        /// Create a bias variable with appropriate initialization
        /// </summary>
        /// <param name="name"></param>
        /// <param name="shape"></param>
        /// <returns></returns>
        private RefVariable bias_variable(string name, int[] shape)
        {
            var initial = tf.constant(0f, shape: shape, dtype: tf.float32);
            return tf.get_variable(name,
                           dtype: tf.float32,
                           initializer: initial);
        }

        /// <summary>
        /// Create a fully-connected layer
        /// </summary>
        /// <param name="x">input from previous layer</param>
        /// <param name="num_units">number of hidden units in the fully-connected layer</param>
        /// <param name="name">layer name</param>
        /// <param name="use_relu">boolean to add ReLU non-linearity (or not)</param>
        /// <returns>The output array</returns>
        private Tensor fc_layer(Tensor x, int num_units, string name, bool use_relu = true)
        {
            return tf_with(tf.variable_scope(name), delegate
            {
                var in_dim = x.shape[1];

                var W = weight_variable("W_" + name, shape: new[] { in_dim, num_units });
                var b = bias_variable("b_" + name, new[] { num_units });

                var layer = tf.matmul(x, W) + b;
                if (use_relu)
                    layer = tf.nn.relu(layer);

                return layer;
            });
        }
        #endregion

        #region Train
        public override void Train()
        {
            // Number of training iterations in each epoch
            var num_tr_iter = (ArrayLabel_Train.Length) / batch_size;

            var init = tf.global_variables_initializer();
            sess.run(init);

            var saver = tf.train.Saver(tf.global_variables(), max_to_keep: 10);

            path_model = Config.Name + "\\MODEL";
            Directory.CreateDirectory(path_model);

            float loss_val = 100.0f;
            float accuracy_val = 0f;

            var sw = new Stopwatch();
            sw.Start();
            foreach (var epoch in range(epochs))
            {
                print($"Training epoch: {epoch + 1}");
                // Randomly shuffle the training data at the beginning of each epoch 
                (ArrayFileName_Train, ArrayLabel_Train) = ShuffleArray(ArrayLabel_Train.Length, ArrayFileName_Train, ArrayLabel_Train);
                y_train = np.eye(Dict_Label.Count)[new NDArray(ArrayLabel_Train)];

                //decay learning rate
                if (learning_rate_step != 0)
                {
                    if ((epoch != 0) && (epoch % learning_rate_step == 0))
                    {
                        learning_rate_base = learning_rate_base * learning_rate_decay;
                        if (learning_rate_base <= learning_rate_min) { learning_rate_base = learning_rate_min; }
                        sess.run(tf.assign(learning_rate, learning_rate_base));
                    }
                }

                //Load local images asynchronously,use queue,improve train efficiency
                BlockingCollection<(NDArray c_x, NDArray c_y, int iter)> BlockC = new BlockingCollection<(NDArray C1, NDArray C2, int iter)>(TrainQueueCapa);
                Task.Run(() =>
                {
                    foreach (var iteration in range(num_tr_iter))
                    {
                        var start = iteration * batch_size;
                        var end = (iteration + 1) * batch_size;
                        (NDArray x_batch, NDArray y_batch) = GetNextBatch(sess, ArrayFileName_Train, y_train, start, end);
                        BlockC.Add((x_batch, y_batch, iteration));
                    }
                    BlockC.CompleteAdding();
                });

                foreach (var item in BlockC.GetConsumingEnumerable())
                {
                    sess.run(optimizer, (x, item.c_x), (y, item.c_y));

                    if (item.iter % display_freq == 0)
                    {
                        // Calculate and display the batch loss and accuracy
                        var result = sess.run(new[] { loss, accuracy }, new FeedItem(x, item.c_x), new FeedItem(y, item.c_y));
                        loss_val = result[0];
                        accuracy_val = result[1];
                        print("CNN：" + ($"iter {item.iter.ToString("000")}: Loss={loss_val.ToString("0.0000")}, Training Accuracy={accuracy_val.ToString("P")} {sw.ElapsedMilliseconds}ms"));
                        sw.Restart();
                    }
                }

                // Run validation after every epoch
                (loss_val, accuracy_val) = sess.run((loss, accuracy), (x, x_valid), (y, y_valid));
                print("CNN：" + "---------------------------------------------------------");
                print("CNN：" + $"gloabl steps: {sess.run(gloabl_steps) },learning rate: {sess.run(learning_rate)}, validation loss: {loss_val.ToString("0.0000")}, validation accuracy: {accuracy_val.ToString("P")}");
                print("CNN：" + "---------------------------------------------------------");

                if (SaverBest)
                {
                    if (accuracy_val > max_accuracy)
                    {
                        max_accuracy = accuracy_val;
                        saver.save(sess, path_model + "\\CNN_Best");
                        print("CKPT Model is save.");
                    }
                }
                else
                {
                    saver.save(sess, path_model + string.Format("\\CNN_Epoch_{0}_Loss_{1}_Acc_{2}", epoch, loss_val, accuracy_val));
                    print("CKPT Model is save.");
                }
            }
            Write_Dictionary(path_model + "\\dic.txt", Dict_Label);
        }

        private void Write_Dictionary(string path, Dictionary<Int64, string> mydic)
        {
            FileStream fs = new FileStream(path, FileMode.Create);
            StreamWriter sw = new StreamWriter(fs);
            foreach (var d in mydic) { sw.Write(d.Key + "," + d.Value + "\r\n"); }
            sw.Flush();
            sw.Close();
            fs.Close();
            print("Write_Dictionary");
        }

        private (NDArray, NDArray) Randomize(NDArray x, NDArray y)
        {
            var perm = np.random.permutation(y.shape[0]);
            np.random.shuffle(perm);
            return (x[perm], y[perm]);
        }

        private (NDArray, NDArray) GetNextBatch(NDArray x, NDArray y, int start, int end)
        {
            var slice = new Slice(start, end);
            var x_batch = x[slice];
            var y_batch = y[slice];
            return (x_batch, y_batch);
        }

        private unsafe (NDArray, NDArray) GetNextBatch(Session sess, string[] x, NDArray y, int start, int end)
        {
            NDArray x_batch = np.zeros(end - start, img_h, img_w, n_channels);
            int n = 0;
            for (int i = start; i < end; i++)
            {
                NDArray img4 = cv2.imread(x[i], IMREAD_COLOR.IMREAD_GRAYSCALE);
                img4 = img4.reshape(img4.shape[0], img4.shape[1], 1);
                x_batch[n] = sess.run(normalized, (decodeJpeg, img4));
                n++;
            }
            var slice = new Slice(start, end);
            var y_batch = y[slice];
            return (x_batch, y_batch);
        }
        #endregion               

        public override void Test()
        {
            (loss_test, accuracy_test) = sess.run((loss, accuracy), (x, x_test), (y, y_test));
            print("CNN：" + "---------------------------------------------------------");
            print("CNN：" + $"Test loss: {loss_test.ToString("0.0000")}, test accuracy: {accuracy_test.ToString("P")}");
            print("CNN：" + "---------------------------------------------------------");

            (Test_Cls, Test_Data) = sess.run((cls_prediction, prob), (x, x_test));
        }

        private void TestDataOutput()
        {
            for (int i = 0; i < ArrayLabel_Test.Length; i++)
            {
                Int64 real = ArrayLabel_Test[i];
                int predict = (int)(Test_Cls[i]);
                var probability = Test_Data[i, predict];
                string result = (real == predict) ? "OK" : "NG";
                string fileName = ArrayFileName_Test[i];
                string real_str = Dict_Label[real];
                string predict_str = Dict_Label[predict];
                print((i + 1).ToString() + "|" + "result:" + result + "|" + "real_str:" + real_str + "|"
                    + "predict_str:" + predict_str + "|" + "probability:" + probability.GetSingle().ToString() + "|"
                    + "fileName:" + fileName);
            }
        }
    }
}
