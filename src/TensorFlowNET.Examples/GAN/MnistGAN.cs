using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Examples.GAN
{
    /// <summary>
    /// https://www.tensorflow.org/tutorials/generative/dcgan
    /// </summary>
    public class MnistGAN : SciSharpExample, IExample
    {
        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "GAN MNIST",
                Enabled = true,
                Priority = 50
            };

        public bool Run()
        {
            var generator = make_generator_model();
            var noise = tf.random.normal((1, 100));
            var generated_image = generator.Apply(noise, training: false);

            return true;
        }

        public Model make_generator_model()
        {
            var model = keras.Sequential();
            model.add(layers.Dense(7 * 7 * 256, use_bias: false, input_shape: 100));
            model.add(layers.BatchNormalization());
            model.add(layers.LeakyReLU());
            model.add(layers.Reshape((7, 7, 256)));

            model.add(layers.Conv2DTranspose(128, (5, 5), strides: (1, 1), padding: "same", use_bias: false));
            model.add(layers.BatchNormalization());
            model.add(layers.LeakyReLU());

            model.add(layers.Conv2DTranspose(64, (5, 5), strides: (2, 2), padding: "same", use_bias: false));
            model.add(layers.BatchNormalization());
            model.add(layers.LeakyReLU());

            model.add(layers.Conv2DTranspose(1, (5, 5), strides: (2, 2), padding: "same", use_bias: false, activation: "tanh"));

            return model;
        }
    }
}
