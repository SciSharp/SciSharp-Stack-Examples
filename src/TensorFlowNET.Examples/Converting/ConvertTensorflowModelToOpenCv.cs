using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow;
using TensorFlowNET.Examples.Utility;
using static Tensorflow.Binding;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Convert tensorflow model to opencv dnn format.
    /// </summary>
    public class ConvertTensorflowModelToOpenCv : IExample
    {
        public bool Enabled { get; set; } = false;
        public bool IsImportingGraph { get; set; } = false;
        public string Name => "Convert TensorFlow Model to OpenCv";

        string modelDir = "ssd_mobilenet_v1_coco_2018_01_28";
        string imageDir = "images";

        string frozen_graph = "frozen_inference_graph.pb";

        public bool Run()
        {
            PrepareData();

            BuildGraph();

            return true;
        }

        public Graph BuildGraph()
        {
            string pb = Path.Combine(modelDir, frozen_graph);
            var graph_def = GraphDef.Parser.ParseFrom(File.ReadAllBytes(pb));

            tf.graph_transforms.TransformGraph(graph_def,
                new[] { "Placeholder" },
                new[] { "Score" },
                new[] { "remove_nodes(op=PlaceholderWithDefault)",
                    "strip_unused_nodes(type=float, shape=\"1,28,28,1\"",
                    "remove_nodes(op=Identity, op=CheckNumerics, op=Switch)",
                    "fold_constants(ignore_errors=true)",
                    "fold_batch_norms",
                    "fold_old_batch_norms",
                    "sort_by_execution_order"});

            /*        with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(new_graph_def.SerializeToString())*/

            throw new NotImplementedException("");
        }

        public Graph ImportGraph()
        {
            throw new NotImplementedException();
        }

        public void Predict(Session sess)
        {
            throw new NotImplementedException();
        }

        public void PrepareData()
        {
            // get model file
            string url = "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz";
            Web.Download(url, modelDir, "ssd_mobilenet_v1_coco.tar.gz");

            Compress.ExtractTGZ(Path.Join(modelDir, "ssd_mobilenet_v1_coco.tar.gz"), "./");

            // download sample picture
            url = $"https://github.com/tensorflow/models/raw/master/research/object_detection/test_images/image2.jpg";
            Web.Download(url, imageDir, "input.jpg");

            // download the pbtxt file
            url = $"https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt";
            Web.Download(url, modelDir, "mscoco_label_map.pbtxt");
        }

        public void Test(Session sess)
        {
            throw new NotImplementedException();
        }

        public void Train(Session sess)
        {
            throw new NotImplementedException();
        }
    }
}
