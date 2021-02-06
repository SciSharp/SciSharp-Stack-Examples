﻿using NumSharp;
using System;
using System.IO;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras.Utils;
using static Tensorflow.Binding;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Inception Architecture for Computer Vision
    /// Port from tensorflow\examples\label_image\label_image.py
    /// </summary>
    public class InceptionArchGoogLeNet : SciSharpExample, IExample
    {
        string dir = "label_image_data";
        string pbFile = "inception_v3_2016_08_28_frozen.pb";
        string labelFile = "imagenet_slim_labels.txt";
        string picFile = "grace_hopper.jpg";
        int input_height = 299;
        int input_width = 299;
        int input_mean = 0;
        int input_std = 255;
        string input_name = "import/input";
        string output_name = "import/InceptionV3/Predictions/Reshape_1";

        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "Inception Arch GoogLeNet",
                Enabled = false,
                IsImportingGraph = false
            };

        public bool Run()
        {
            PrepareData();

            var labels = File.ReadAllLines(Path.Join(dir, labelFile));

            var nd = ReadTensorFromImageFile(Path.Join(dir, picFile),
                input_height: input_height,
                input_width: input_width,
                input_mean: input_mean,
                input_std: input_std);

            var graph = new Graph();
            graph.Import(Path.Join(dir, pbFile));
            var input_operation = graph.get_operation_by_name(input_name);
            var output_operation = graph.get_operation_by_name(output_name);

            NDArray results;
            using (var sess = tf.Session(graph))
            {
                results = sess.run(output_operation.outputs[0],
                    new FeedItem(input_operation.outputs[0], nd));
            }

            results = np.squeeze(results);

            var argsort = results.argsort<float>();
            var top_k = argsort.Data<float>()
                .Skip(results.size - 5)
                .Reverse()
                .ToArray();

            foreach (float idx in top_k)
                Console.WriteLine($"{picFile}: {idx} {labels[(int)idx]}, {results[(int)idx]}");

            return true;
        }

        private NDArray ReadTensorFromImageFile(string file_name,
                                int input_height = 299,
                                int input_width = 299,
                                int input_mean = 0,
                                int input_std = 255)
        {
            var graph = tf.Graph().as_default();

            var file_reader = tf.io.read_file(file_name, "file_reader");
            var image_reader = tf.image.decode_jpeg(file_reader, channels: 3, name: "jpeg_reader");
            var caster = tf.cast(image_reader, tf.float32);
            var dims_expander = tf.expand_dims(caster, 0);
            var resize = tf.constant(new int[] { input_height, input_width });
            var bilinear = tf.image.resize_bilinear(dims_expander, resize);
            var sub = tf.subtract(bilinear, new float[] { input_mean });
            var normalized = tf.divide(sub, new float[] { input_std });

            using (var sess = tf.Session(graph))
                return sess.run(normalized);
        }

        public override void PrepareData()
        {
            Directory.CreateDirectory(dir);

            // get model file
            string url = "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz";

            Web.Download(url, dir, $"{pbFile}.tar.gz");

            Compress.ExtractTGZ(Path.Join(dir, $"{pbFile}.tar.gz"), dir);

            // download sample picture
            string pic = "grace_hopper.jpg";
            url = $"https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/label_image/data/{pic}";
            Web.Download(url, dir, pic);
        }
    }
}
