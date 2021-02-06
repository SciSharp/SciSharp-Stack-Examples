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
            return true;
        }

        public override void BuildModel()
        {
            var model = keras.Sequential();
            model.add(layers.Dense(7 * 7 * 256, use_bias: false, input_shape: 100));
            model.add(layers.BatchNormalization());
            model.add(layers.LeakyReLU());
            model.add(layers.Reshape((7, 7, 256)));

            // model.add(layers.Conv2DTranspose(128, (5, 5), strides: (1, 1), padding: "same", use_bias: false));
            model.add(layers.BatchNormalization());
            model.add(layers.LeakyReLU());

            // model.add(layers.Conv2DTranspose(64, (5, 5), strides: (2, 2), padding: "same", use_bias: false));
            model.add(layers.BatchNormalization());
            model.add(layers.LeakyReLU());

            // model.add(layers.Conv2DTranspose(1, (5, 5), strides: (2, 2), padding: "same", use_bias: false, activation: "tanh"));
        }
    }
}
