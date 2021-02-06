using System.IO;
using Tensorflow.Keras.Utils;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/text_classification_from_scratch.ipynb#scrollTo=qqTCrB7SmJv9
    /// </summary>
    public class CnnTextClassificationKeras : SciSharpExample, IExample
    {
        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "CNN Text Classification (Keras)",
                Enabled = false
            };

        public bool Run()
        {
            return true;
        }

        public override void PrepareData()
        {
            string fileName = "aclImdb_v1.tar.gz";
            string url = $"https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz";
            string data_dir = Path.GetTempPath();
            Web.Download(url, data_dir, fileName);
            Compress.ExtractGZip(Path.Join(data_dir, fileName), data_dir);
            data_dir = Path.Combine(data_dir, "aclImdb_v1");
        }
    }
}
