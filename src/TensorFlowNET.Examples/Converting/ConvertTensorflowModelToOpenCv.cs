using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow;
using TensorFlowNET.Examples.Utility;
using static Tensorflow.Binding;
using Google.Protobuf;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Convert tensorflow model to opencv dnn format.
    /// </summary>
    public class ConvertTensorflowModelToOpenCv : SciSharpExample, IExample
    {
        string modelDir = "ConvertTfModelToOpenCv";
        string imageDir = "images";

        string frozen_graph = "frozen_graph.pb";
        string output_graph = "output_graph.pb";

        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "Convert TensorFlow Model to OpenCv",
                Enabled = true,
                IsImportingGraph = false
            };

        public bool Run()
        {
            PrepareData();

            BuildGraph();

            return true;
        }

        public override Graph BuildGraph()
        {
            string pb = Path.Combine(modelDir, frozen_graph);
            var graph_def = GraphDef.Parser.ParseFrom(File.ReadAllBytes(pb));

            // transform_graph
            graph_def = tf.graph_transforms.TransformGraph(graph_def,
                new[] { "Placeholder" },
                new[] { "Score" },
                new[] { "remove_nodes(op=PlaceholderWithDefault)",
                    "strip_unused_nodes(type=float, shape=\"1,28,28,1\")",
                    "remove_nodes(op=Identity, op=CheckNumerics, op=Switch)",
                    "fold_constants(ignore_errors=true)",
                    "fold_batch_norms",
                    "fold_old_batch_norms",
                    "sort_by_execution_order"});

            // convert_to_constant
            var keep_prob = tf.constant(1.0f, dtype: tf.float32, shape: new int[0], name: "keep_prob");
            var weight_factor = tf.constant(1.0f, dtype: tf.float32, shape: new int[0], name: "weight_factor");
            var is_training = tf.constant(false, dtype: tf.@bool, shape: new int[0], name: "is_training");

            var new_graph_def = new GraphDef();
            foreach (var node in graph_def.Node)
            {
                switch (node.Name)
                {
                    case "keep_prob":
                        new_graph_def.Node.Add(keep_prob.op.node_def);
                        break;
                    case "weight_factor":
                        new_graph_def.Node.Add(weight_factor.op.node_def);
                        break;
                    case "is_training":
                        new_graph_def.Node.Add(is_training.op.node_def);
                        break;
                    default:
                        new_graph_def.Node.Add(node.Clone());
                        break;
                }
            }

            // optimize_batch_normalization
            graph_def = new_graph_def;
            new_graph_def = new GraphDef();
            foreach (var node in graph_def.Node)
            {
                var modified_node = node.Clone();
                if (node.Name.StartsWith("conv"))
                {

                }
                else if(node.Name.StartsWith("fc") || node.Name.StartsWith("logits"))
                {

                }

                new_graph_def.Node.Add(modified_node);
            }

            // transform_graph
            new_graph_def = tf.graph_transforms.TransformGraph(graph_def,
                new[] { "Placeholder" },
                new[] { "Score" },
                new[] { "remove_nodes(op=PlaceholderWithDefault)",
                    "strip_unused_nodes(type=float, shape=\"1,28,28,1\")",
                    "remove_nodes(op=Identity, op=CheckNumerics, op=Switch)",
                    "fold_constants(ignore_errors=true)",
                    "fold_batch_norms",
                    "fold_old_batch_norms",
                    "sort_by_execution_order"});

            // remove_dropout
            graph_def = new_graph_def;
            new_graph_def = new GraphDef();
            foreach (var node in graph_def.Node)
            {
                var modified_node = node.Clone();
                if (node.Name.StartsWith("dropout1") || node.Name.StartsWith("dropout2"))
                {
                    continue;
                }
                
                if (node.Name == "fc2/fc2/batch_norm/batchnorm/mul_1")
                {
                    modified_node.Input[0] = "mul";
                    modified_node.Input[1] = "fc2/weights";
                }

                if (node.Name == "logits/logits/batch_norm/batchnorm/mul_1")
                {
                    modified_node.Input[0] = "fc2/activation";
                    modified_node.Input[1] = "logits/weights";
                }

                new_graph_def.Node.Add(modified_node);
            }

            // save the graph
            string output_pb = Path.Combine(modelDir, output_graph);
            File.WriteAllBytes(output_pb, new_graph_def.ToByteArray());
            return null;
        }

        public override void PrepareData()
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
    }
}
