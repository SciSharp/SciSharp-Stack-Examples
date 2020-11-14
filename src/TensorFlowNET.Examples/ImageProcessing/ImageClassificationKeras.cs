using System.Collections.Generic;
using Tensorflow;
using Tensorflow.Keras;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// This tutorial shows how to classify images of flowers.
    /// https://www.tensorflow.org/tutorials/images/classification
    /// </summary>
    public class ImageClassificationKeras : SciSharpExample, IExample
    {
        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "Image Classification (Keras)",
                Enabled = true,
                Priority = 18
            };

        public bool Run()
        {
            PrepareData();
            return true;
        }

        public override void PrepareData()
        {
            int batch_size = 32;
            TensorShape img_dim = (180, 180);

            var data_dir = @"C:/Users/haipi/.keras/datasets/flower_photos";
            var train_ds = keras.preprocessing.image_dataset_from_directory(data_dir,
                validation_split: 0.2f,
                subset: "training",
                seed: 123,
                image_size: img_dim,
                batch_size: batch_size);

            var val_ds = keras.preprocessing.image_dataset_from_directory(data_dir,
                validation_split: 0.2f,
                subset: "validation",
                seed: 123,
                image_size: img_dim,
                batch_size: batch_size);

            train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size: -1);
            val_ds = val_ds.cache().prefetch(buffer_size: -1);

            foreach (var (img, label) in train_ds)
            {
                print("batch images: " + img.TensorShape);
                print("labels: " + label);
            }

            int num_classes = 5;
            // var normalization_layer = tf.keras.layers.Rescaling(1.0f / 255);
            var layers = keras.layers;
            var model = keras.Sequential(new List<ILayer>
            {
                layers.Rescaling(1.0f / 255, input_shape: (img_dim.dims[0], img_dim.dims[1], 3)),
                layers.Conv2D(16, 3, padding: "same", activation: keras.activations.Relu),
                layers.MaxPooling2D(),
                /*layers.Conv2D(32, 3, padding: "same", activation: "relu"),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, padding: "same", activation: "relu"),
                layers.MaxPooling2D(),*/
                layers.Flatten(),
                layers.Dense(128, activation: keras.activations.Relu),
                layers.Dense(num_classes)
            });

            model.compile("adam", keras.losses.SparseCategoricalCrossentropy(from_logits: true));
        }
    }
}
