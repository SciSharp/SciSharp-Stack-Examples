using System;
using System.IO;
using Tensorflow;
using TensorFlowNET.Examples.Utility;
using static Tensorflow.Binding;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// This example removes the background from an input image.
    /// 
    /// https://github.com/susheelsk/image-background-removal
    /// </summary>
    public class ImageBackgroundRemoval : SciSharpExample, IExample
    {
        string dataDir = "deeplabv3";
        string modelDir = "deeplabv3_mnv2_pascal_train_aug";
        string modelName = "frozen_inference_graph.pb";

        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "Image Background Removal",
                Enabled = false,
                IsImportingGraph = true
            };

        public bool Run()
        {
            PrepareData();

            // import GraphDef from pb file
            var graph = new Graph().as_default();
            graph.Import(Path.Join(dataDir, modelDir, modelName));

            Tensor output = graph.OperationByName("SemanticPredictions");

            using (var sess = tf.Session(graph))
            {
                // Runs inference on a single image.
                sess.run(output, new FeedItem(output, "[np.asarray(resized_image)]"));
            }

            return false;
        }

        public override void PrepareData()
        {
            // get mobile_net_model file
            string fileName = "deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz";
            string url = $"http://download.tensorflow.org/models/{fileName}";
            Web.Download(url, dataDir, fileName);
            Compress.ExtractTGZ(Path.Join(dataDir, fileName), dataDir);

            // xception_model, better accuracy
            /*fileName = "deeplabv3_pascal_train_aug_2018_01_04.tar.gz";
            url = $"http://download.tensorflow.org/models/{fileName}";
            Web.Download(url, modelDir, fileName);
            Compress.ExtractTGZ(Path.Join(modelDir, fileName), modelDir);*/
        }
    }
}
