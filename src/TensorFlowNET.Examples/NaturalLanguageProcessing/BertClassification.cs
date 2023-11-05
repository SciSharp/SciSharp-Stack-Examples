using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow.Operations.Initializers;
using static Tensorflow.KerasApi;
using BERT;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.Keras.Engine.InputSpec;

namespace TensorFlowNET.Examples
{
    class BertClassification : SciSharpExample, IExample
    {
        int max_seq_len = 512;
        int batch_size = 32;
        int num_classes = 2;
        int epoch = 10;
        float learning_rate = (float)2e-5;
        string pretrained_weight_path = "./tf_model.h5";
        BertConfig config = new BertConfig();
        NDArray np_x_train;
        NDArray np_y_train;
        public ExampleConfig InitConfig()
        => Config = new ExampleConfig
        {
            Name = "Bert for Classification",
            Enabled = false
        };

        public override void PrepareData()
        {
            // tf.debugging.set_log_device_placement(true);
            Console.WriteLine("Preparing data...");
            string url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz";
            var dataset = keras.utils.get_file("aclImdb_v1.tar.gz", url,
                untar: true,
                cache_dir: Path.GetTempPath(),
                cache_subdir: "aclImdb_v1");
            var data_dir = Path.Combine(dataset, "aclImdb");
            var train_dir = Path.Combine(data_dir, "train");
            (int[,] x_train_neg, int[] y_train_neg) = IMDBDataPreProcessor.
                    ProcessData(Path.Combine(train_dir, "neg"), max_seq_len, 0);
            (int[,] x_train_pos, int[] y_train_pos) = IMDBDataPreProcessor.
                    ProcessData(Path.Combine(train_dir, "pos"), max_seq_len, 1);
            np_x_train = np.array(x_train_neg, dtype: tf.int32);
            np_y_train = np.array(y_train_neg, dtype: tf.int32);
            np_x_train = np.concatenate((np_x_train, np.array(x_train_pos, dtype: tf.int32)), 0);
            np_y_train = np.concatenate((np_y_train, np.array(y_train_pos, dtype: tf.int32)), 0);
        }

        public bool Run()
        {
            var model = keras.Sequential();
            model.add(keras.layers.Input(max_seq_len, batch_size, dtype: tf.int32));
            model.add(new BertMainLayer(config));
            if(File.Exists(pretrained_weight_path)) model.load_weights(pretrained_weight_path);
            model.add(keras.layers.Dense(num_classes));
            model.compile(optimizer: keras.optimizers.AdamW(learning_rate, weight_decay: 0.01f, no_decay_params: new List<string> { "gamma", "beta" }),
                loss: keras.losses.SparseCategoricalCrossentropy(from_logits: true), metrics: new[] { "acc" });
            model.summary();
            PrepareData();
            model.fit(np_x_train, np_y_train,
                batch_size: batch_size,
                epochs: epoch,
                shuffle: true,
                validation_split: 0.2f);
            return true;
        }
    }
}
