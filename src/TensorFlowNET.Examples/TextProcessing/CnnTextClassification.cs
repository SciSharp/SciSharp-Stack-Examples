﻿/*****************************************************************************
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
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras.Utils;
using Tensorflow.Sessions;
using TensorFlowNET.Examples.Text;
using static Tensorflow.Binding;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// https://github.com/dongjun-Lee/text-classification-models-tf
    /// </summary>
    public class CnnTextClassification : SciSharpExample, IExample
    {
        public int? DataLimit = null;

        const string dataDir = "cnn_text";

        string TRAIN_PATH = $"{dataDir}/dbpedia_csv/train.csv";

        int NUM_CLASS = 14;
        int BATCH_SIZE = 64;
        int NUM_EPOCHS = 10;
        int WORD_MAX_LEN = 100;
        int CHAR_MAX_LEN = 1014;

        float loss_value = 0;
        double max_accuracy = 0;

        int alphabet_size = -1;
        int vocabulary_size = -1;
        NDArray train_x, test_x, train_y, test_y;

        ITextModel textModel;
        public string ModelName = "word_cnn"; // word_cnn | char_cnn | vd_cnn | word_rnn | att_rnn | rcnn

        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "CNN Text Classification (Graph)",
                Enabled = true,
                IsImportingGraph = false
            };

        public bool Run()
        {
            tf.compat.v1.disable_eager_execution();

            PrepareData();
            Predict();
            Test();
            Train();
            FreezeModel();

            return max_accuracy > 0.9;
        }

        // TODO: this originally is an SKLearn utility function. it randomizes train and test which we don't do here
        private (NDArray, NDArray, NDArray, NDArray) train_test_split(NDArray x, NDArray y, float test_size = 0.3f)
        {
            Console.WriteLine("Splitting in Training and Testing data...");
            int len = x.shape[0];
            //int classes = y.Data<int>().Distinct().Count();
            //int samples = len / classes;
            int train_size = (int)Math.Round(len * (1 - test_size));
            train_x = x[new Slice(stop: train_size), new Slice()];
            test_x = x[new Slice(start: train_size), new Slice()];
            train_y = y[new Slice(stop: train_size)];
            test_y = y[new Slice(start: train_size)];
            Console.WriteLine("\tDONE");

            return (train_x, test_x, train_y, test_y);
        }

        private void FillWithShuffledLabels(int[][] x, int[] y, int[][] shuffled_x, int[] shuffled_y, Random random, Dictionary<int, HashSet<int>> labels)
        {
            int i = 0;
            var label_keys = labels.Keys.ToArray();
            while (i < shuffled_x.Length)
            {
                var key = label_keys[random.Next(label_keys.Length)];
                var set = labels[key];
                var index = set.First();
                if (set.Count == 0)
                {
                    labels.Remove(key); // remove the set as it is empty
                    label_keys = labels.Keys.ToArray();
                }
                shuffled_x[i] = x[index];
                shuffled_y[i] = y[index];
                i++;
            }
        }

        private IEnumerable<(NDArray, NDArray, int)> batch_iter(NDArray inputs, NDArray outputs, int batch_size, int num_epochs)
        {
            var num_batches_per_epoch = (len(inputs) - 1) / batch_size + 1;
            var total_batches = num_batches_per_epoch * num_epochs;
            foreach (var epoch in range(num_epochs))
            {
                foreach (var batch_num in range(num_batches_per_epoch))
                {
                    var start_index = batch_num * batch_size;
                    var end_index = Math.Min((batch_num + 1) * batch_size, len(inputs));
                    if (end_index <= start_index)
                        break;
                    yield return (inputs[new Slice(start_index, end_index)], outputs[new Slice(start_index, end_index)], total_batches);
                }
            }
        }

        public override void PrepareData()
        {
            // full dataset https://github.com/le-scientifique/torchDatasets/raw/master/dbpedia_csv.tar.gz
            var url = "https://raw.githubusercontent.com/SciSharp/TensorFlow.NET/master/data/dbpedia_subset.zip";
            Web.Download(url, dataDir, "dbpedia_subset.zip");
            Compress.UnZip(Path.Combine(dataDir, "dbpedia_subset.zip"), Path.Combine(dataDir, "dbpedia_csv"));

            Console.WriteLine("Building dataset...");
            var (x, y) = (new int[0][], new int[0]);

            if (ModelName == "char_cnn")
            {
                (x, y, alphabet_size) = DataHelpers.build_char_dataset(TRAIN_PATH, "char_cnn", CHAR_MAX_LEN);
            }
            else
            {
                var word_dict = DataHelpers.build_word_dict(TRAIN_PATH);
                vocabulary_size = len(word_dict);
                (x, y) = DataHelpers.build_word_dataset(TRAIN_PATH, word_dict, WORD_MAX_LEN);
            }

            Console.WriteLine("\tDONE ");

            (train_x, test_x, train_y, test_y) = train_test_split(x, y, test_size: 0.15f);
            Console.WriteLine("Training set size: " + train_x.shape[0]);
            Console.WriteLine("Test set size: " + test_x.shape[0]);
        }

        public override Graph ImportGraph()
        {
            var graph = tf.Graph().as_default();

            // download graph meta data
            var meta_file = "word_cnn.meta";
            var meta_path = Path.Combine("graph", meta_file);
            if (File.GetLastWriteTime(meta_path) < new DateTime(2019, 05, 11))
            {
                // delete old cached file which contains errors
                Console.WriteLine("Discarding cached file: " + meta_path);
                if (File.Exists(meta_path))
                    File.Delete(meta_path);
            }
            var url = "https://raw.githubusercontent.com/SciSharp/TensorFlow.NET/master/graph/" + meta_file;
            Web.Download(url, "graph", meta_file);

            Console.WriteLine("Import graph...");
            tf.train.import_meta_graph(Path.Join("graph", meta_file));
            Console.WriteLine("\tDONE ");

            return graph;
        }

        public override Graph BuildGraph()
        {
            var graph = tf.Graph().as_default();

            switch (ModelName)
            {
                case "word_cnn":
                    textModel = new WordCnn(vocabulary_size, WORD_MAX_LEN, NUM_CLASS);
                    break;
                case "char_cnn":
                    textModel = new CharCnn(alphabet_size, CHAR_MAX_LEN, NUM_CLASS);
                    break;
            }

            return graph;
        }

        public override void Train()
        {
            var graph = Config.IsImportingGraph ? ImportGraph() : BuildGraph();

            using (var sess = tf.Session(graph))
            {
                sess.run(tf.global_variables_initializer());
                var saver = tf.train.Saver(tf.global_variables());

                var train_batches = batch_iter(train_x, train_y, BATCH_SIZE, NUM_EPOCHS);
                var num_batches_per_epoch = (len(train_x) - 1) / BATCH_SIZE + 1;

                Tensor is_training = graph.OperationByName("is_training");
                Tensor model_x = graph.OperationByName("x");
                Tensor model_y = graph.OperationByName("y");
                Tensor loss = graph.OperationByName("loss/Mean");
                Operation optimizer = graph.OperationByName("loss/Adam");
                Tensor global_step = graph.OperationByName("Variable");
                Tensor accuracy = graph.OperationByName("accuracy/accuracy");

                var sw = new Stopwatch();
                sw.Start();

                int step = 0;
                foreach (var (x_batch, y_batch, total) in train_batches)
                {
                    (_, step, loss_value) = sess.run((optimizer, global_step, loss),
                        (model_x, x_batch), (model_y, y_batch), (is_training, true));
                    if (step % 10 == 0)
                    {
                        Console.WriteLine($"Training on batch {step}/{total} loss: {loss_value.ToString("0.0000")} {sw.ElapsedMilliseconds}ms.");
                        sw.Restart();
                    }

                    if (step % 100 == 0)
                    {
                        // Test accuracy with validation data for each epoch.
                        var valid_batches = batch_iter(test_x, test_y, BATCH_SIZE, 1);
                        var (sum_accuracy, cnt) = (0.0f, 0);
                        foreach (var (valid_x_batch, valid_y_batch, total_validation_batches) in valid_batches)
                        {
                            var valid_feed_dict = new FeedDict
                            {
                                [model_x] = valid_x_batch,
                                [model_y] = valid_y_batch,
                                [is_training] = false
                            };
                            float accuracy_value = sess.run(accuracy, (model_x, valid_x_batch), (model_y, valid_y_batch), (is_training, false));
                            sum_accuracy += accuracy_value;
                            cnt += 1;
                        }

                        var valid_accuracy = sum_accuracy / cnt;

                        print($"\nValidation Accuracy = {valid_accuracy.ToString("P")}\n");

                        // Save model
                        if (valid_accuracy > max_accuracy)
                        {
                            max_accuracy = valid_accuracy;
                            saver.save(sess, $"{dataDir}/word_cnn.ckpt", global_step: step);
                            print("Model is saved.\n");
                        }
                    }
                }
            }
        }

        public override void Test()
        {
            var checkpoint = Path.Combine(dataDir, "word_cnn.ckpt-800");
            if (!File.Exists($"{checkpoint}.meta")) return;

            var graph = tf.Graph();
            using (var sess = tf.Session(graph))
            {
                var saver = tf.train.import_meta_graph($"{checkpoint}.meta");
                saver.restore(sess, checkpoint);

                Tensor x = graph.get_operation_by_name("x");
                Tensor y = graph.get_operation_by_name("y");
                Tensor is_training = graph.get_operation_by_name("is_training");
                Tensor accuracy = graph.get_operation_by_name("accuracy/accuracy");

                var batches = batch_iter(test_x, test_y, BATCH_SIZE, 1);
                float sum_accuracy = 0;
                int cnt = 0;
                foreach (var (batch_x, batch_y, total) in batches)
                {
                    float accuracy_out = sess.run(accuracy, (x, batch_x), (y, batch_y), (is_training, false));
                    sum_accuracy += accuracy_out;
                    cnt += 1;
                }
                print($"Test Accuracy : {sum_accuracy / cnt}");
            }
        }

        public override void Predict()
        {
            var model = Path.Combine(dataDir, "frozen_model.pb");
            if (!File.Exists(model)) return;

            var graph = tf.train.load_graph(model);
            using (var sess = tf.Session(graph))
            {
                Tensor x = graph.get_operation_by_name("x");
                Tensor is_training = graph.get_operation_by_name("is_training");
                Tensor prediction = graph.get_operation_by_name("output/ArgMax");
                // encode text into 100 dimensions
                var batches = batch_iter(test_x, test_y, BATCH_SIZE, 1).First();
                var input = batches.Item1[0].reshape(1, 100);
                var result = sess.run(prediction, (x, input), (is_training, false));
            }
        }

        public override string FreezeModel()
        {
            return tf.train.freeze_graph(dataDir,
                "frozen_model",
                new[] { "output/ArgMax" });
        }
    }
}
