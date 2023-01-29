using SciSharp.Models.TimeSeries;
using SciSharp.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Examples;

public class SentimentClassification : SciSharpExample, IExample
{
    ITimeSeriesTask task;
    public ExampleConfig InitConfig()
        => Config = new ExampleConfig
        {
            Name = "Text Sentiment Classification",
            Enabled = true
        };

    public bool Run()
    {
        var wizard = new ModelWizard();
        task = wizard.AddTimeSeriesTask<ConvolutionalModel>(new TaskOptions
        {
            WeightsPath = @"timeseries_linear_v1\saved_weights.h5"
        });
        task.SetModelArgs(new TimeSeriesModelArgs
        {
        });

        return true;
    }

    public override void PrepareData()
    {
        // tf.debugging.set_log_device_placement(true);
        string url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz";
        var dataset = keras.utils.get_file("aclImdb_v1.tar.gz", url,
            untar: true,
            cache_dir: Path.GetTempPath(),
            cache_subdir: "aclImdb_v1");
        var data_dir = Path.Combine(dataset, "aclImdb");
        var train_dir = Path.Combine(data_dir, "train");

        int batch_size = 32;
        int seed = 42;
        var raw_train_ds = keras.preprocessing.text_dataset_from_directory(
            train_dir,
            batch_size: batch_size,
            validation_split: 0.2f,
            subset: "training",
            seed: seed);

        /*foreach (var (text_batch, label_batch) in raw_train_ds.take(1))
        {
            foreach (var i in range(3))
            {
                print("Review", text_batch.StringData()[i]);
                print("Label", label_batch.numpy()[i]);
            }
        }

        print("Label 0 corresponds to", raw_train_ds.class_names[0]);
        print("Label 1 corresponds to", raw_train_ds.class_names[1]);

        var raw_val_ds = keras.preprocessing.text_dataset_from_directory(
            train_dir,
            batch_size: batch_size,
            validation_split: 0.2f,
            subset: "validation",
        seed: seed);

        var test_dir = Path.Combine(data_dir, "test");
        var raw_test_ds = keras.preprocessing.text_dataset_from_directory(
            test_dir,
            batch_size: batch_size);

        var max_features = 10000;
        var sequence_length = 250;

        Func<Tensor, Tensor> custom_standardization = input_data =>
        {
            var lowercase = tf.strings.lower(input_data);
            var stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ");
            return tf.strings.regex_replace(stripped_html,
                                            "'[!\"\\#\\$%\\&\'\\(\\)\\*\\+,\\-\\./:;<=>\\?@\\[\\\\\\]\\^_`\\{\\|\\}\\~]'",
            "");
        };

        var vectorize_layer = keras.layers.preprocessing.TextVectorization(standardize: custom_standardization,
            max_tokens: max_features,
            output_mode: "int",
            output_sequence_length: sequence_length);

        var train_text = raw_train_ds.map(inputs => inputs[0]);
        //vectorize_layer.adapt(train_text);*/
    }
}
