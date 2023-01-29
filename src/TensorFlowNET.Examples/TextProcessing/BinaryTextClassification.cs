using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Examples;

/// <summary>
/// This tutorial demonstrates text classification starting from plain text files stored on disk.
/// You'll train a binary classifier to perform sentiment analysis on an IMDB dataset. 
/// At the end of the notebook, there is an exercise for you to try, in which you'll train a 
/// multiclass classifier to predict the tag for a programming question on Stack Overflow.
/// https://www.tensorflow.org/tutorials/keras/text_classification
/// </summary>
public class BinaryTextClassification : SciSharpExample, IExample
{
    NDArray train_data, train_labels, test_data, test_labels;

    public ExampleConfig InitConfig()
        => Config = new ExampleConfig
        {
            Name = "Binary Text Classification",
            Enabled = false
        };

    public bool Run()
    {
        PrepareData();

        // Console.WriteLine($"Training entries: {train_data.shape[0]}, labels: {train_labels.shape[0]}");

        // A dictionary mapping words to an integer index
        /*train_data = keras.preprocessing.sequence.pad_sequences(train_data,
            value: word_index["<PAD>"],
            padding: "post",
            maxlen: 256);

        test_data = keras.preprocessing.sequence.pad_sequences(test_data,
            value: word_index["<PAD>"],
            padding: "post",
            maxlen: 256);*/

        // input shape is the vocabulary count used for the movie reviews (10,000 words)

        var model = keras.Sequential();
        //var layer = tf.keras.layers.Embedding(vocab_size, 16);
        //model.add(layer);

        return false;
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

        foreach (var (text_batch, label_batch) in raw_train_ds.take(1))
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
        //vectorize_layer.adapt(train_text);
    }
}
