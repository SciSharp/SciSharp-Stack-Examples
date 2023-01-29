using System;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Datasets;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow.NumPy;
using System.Drawing;
using System.IO;

namespace TensorFlowNET.Examples;

/// <summary>
/// https://www.tensorflow.org/tutorials/generative/dcgan
/// AtCode:JG5FLDRWHY9FEZ9S V559.83530 Provided by big crabs
/// </summary>
public class MnistGAN : SciSharpExample, IExample
{
    float LeakyReLU_alpha = 0.2f;

#if GPU
    int epochs = 5000; //1000; // Better effect, but longer time
    int batch_size = 256; //16;
#else
    int epochs = 1000; //20;
    int batch_size = 64;
#endif

    string imgpath = "dcgan\\imgs";
    string modelpath = "dcgan\\models";
    Shape img_shape;
    int latent_dim = 100;
    int img_rows = 28;
    int img_cols = 28;
    int channels = 1;

    DatasetPass data;

    public ExampleConfig InitConfig()
        => Config = new ExampleConfig
        {
            Name = "GAN MNIST",
            Enabled = true
        };

    public bool Run()
    {
        tf.enable_eager_execution();

        PrepareData();
        Train();
        //Test();

        return true;
    }

    public override void PrepareData()
    {
        data = keras.datasets.mnist.load_data();

        img_shape = (img_rows, img_cols, channels);
        if (img_cols % 4 != 0 || img_rows % 4 != 0)
        {
            throw new Exception("The width and height of the image must be a multiple of 4");
        }
        Directory.CreateDirectory(imgpath);
        Directory.CreateDirectory(modelpath);
    }
    private Model Make_Generator_model()
    {
        Tensorflow.Keras.Activation activation = null;

        var model = keras.Sequential();
        model.add(keras.layers.Input(shape: 100));
        model.add(keras.layers.Dense(784, activation: activation));
        model.add(keras.layers.LeakyReLU(LeakyReLU_alpha));
        model.add(keras.layers.Dense(784, activation: activation));
        model.add(keras.layers.LeakyReLU(LeakyReLU_alpha));
        model.add(keras.layers.Reshape((28, 28, 1)));
        model.add(keras.layers.Tanh());
        model.summary();
        return model;


        //Tensorflow.Keras.Activation activation = null;
        //var model = keras.Sequential();
        //model.add(keras.layers.Input(shape: 100));
        //model.add(keras.layers.Dense(img_rows / 4 * img_cols / 4 * 256, activation: activation));
        //model.add(keras.layers.BatchNormalization(momentum: 0.8f));
        //model.add(keras.layers.LeakyReLU(LeakyReLU_alpha));
        //model.add(keras.layers.Reshape((7, 7, 256)));
        //model.add(keras.layers.UpSampling2D());
        //model.add(keras.layers.Conv2D(128, 3, 1, padding: "same", activation: activation));
        //model.add(keras.layers.BatchNormalization(momentum: 0.8f));
        //model.add(keras.layers.LeakyReLU(LeakyReLU_alpha));
        //model.add(keras.layers.UpSampling2D());
        //model.add(keras.layers.Conv2D(64, 3, 1, padding: "same", activation: activation));
        //model.add(keras.layers.BatchNormalization(momentum: 0.8f));
        //model.add(keras.layers.LeakyReLU(LeakyReLU_alpha));
        //model.add(keras.layers.Conv2D(32, 3, 1, padding: "same", activation: activation));
        //model.add(keras.layers.BatchNormalization(momentum: 0.8f));
        //model.add(keras.layers.LeakyReLU(LeakyReLU_alpha));
        //model.add(keras.layers.Conv2D(1, 3, 1, padding: "same", activation: "tanh"));
        //model.summary();
        //return model;
    }

    private Model Make_Discriminator_model()
    {
        Tensorflow.Keras.Activation activation = null;
        var image = keras.Input(img_shape);
        var x = keras.layers.Reshape(784).Apply(image);
        x = keras.layers.Dense(784, activation: activation).Apply(x);
        x = keras.layers.LeakyReLU(LeakyReLU_alpha).Apply(x);
        x = keras.layers.Dense(784, activation: activation).Apply(x);
        x = keras.layers.LeakyReLU(LeakyReLU_alpha).Apply(x);
        x = keras.layers.Dense(1, activation: "sigmoid").Apply(x);
        var model = keras.Model(image, x);
        model.summary();
        return model;


        //Tensorflow.Keras.Activation activation = null;
        //var image = keras.Input(img_shape);
        //var x = keras.layers.Conv2D(128, kernel_size: 3, strides: (2, 2), padding: "same", activation: activation).Apply(image);
        //x = keras.layers.LeakyReLU(LeakyReLU_alpha).Apply(x);
        //x = keras.layers.BatchNormalization(momentum: 0.8f).Apply(x);
        //x = keras.layers.Conv2D(256, 3, (2, 2), "same", activation: activation).Apply(x);
        //x = keras.layers.BatchNormalization(momentum: 0.8f).Apply(x);
        //x = keras.layers.LeakyReLU(LeakyReLU_alpha).Apply(x);
        //x = keras.layers.Conv2D(512, 3, (2, 2), "same", activation: activation).Apply(x);
        //x = keras.layers.BatchNormalization(momentum: 0.8f).Apply(x);
        //x = keras.layers.LeakyReLU(LeakyReLU_alpha).Apply(x);
        //x = keras.layers.Conv2D(1024, 3, (2, 2), "same", activation: activation).Apply(x);
        //x = keras.layers.BatchNormalization(momentum: 0.8f).Apply(x);
        //x = keras.layers.LeakyReLU(LeakyReLU_alpha).Apply(x);
        //x = keras.layers.Flatten().Apply(x);
        //x = keras.layers.Dense(1, activation: "sigmoid").Apply(x);
        //var model = keras.Model(image, x);
        //model.summary();
        //return model;
    }

    public override void Train()
    {
        NDArray X_train = data.Train.Item1; // [60000,28,28]
        X_train = X_train / 127.5 - 1;
        X_train = np.expand_dims(X_train, 3);
        X_train = X_train.astype(np.float32);

        var G = Make_Generator_model();
        var D = Make_Discriminator_model();

        float d_lr = 2e-4f;
        float g_lr = 2e-4f;
        var d_optimizer = keras.optimizers.Adam(d_lr, 0.5f);
        var g_optimizer = keras.optimizers.Adam(g_lr, 0.5f);
        int showstep = 10;

        for (var i = 0; i <= epochs; i++)
        {
            while (File.Exists("dcgan\\models\\Model_" + (i + 100) + "_g.weights") && File.Exists("dcgan\\models\\Model_" + (i + 100) + "_d.weights"))
                i += 100;
            if (File.Exists("dcgan\\models\\Model_" + i + "_g.weights") && File.Exists("dcgan\\models\\Model_" + i + "_d.weights"))
            {
                Console.WriteLine("Loading weights for epoch " + i);
                G.load_weights("dcgan\\models\\Model_" + i + "_g.weights");
                D.save_weights("dcgan\\models\\Model_" + i + "_d.weights");
                PredictImage(G, i);
            }
            else
            {
                var idx = np.random.randint(0, (int)X_train.shape[0], size: batch_size);
                var imgs = X_train[idx];

                Tensor g_loss, d_loss, d_loss_real, d_loss_fake;
                using (var tape = tf.GradientTape(true))
                {
                    var noise = np.random.normal(0, 1, new int[] { batch_size, 100 });
                    noise = noise.astype(np.float32);
                    var noise_z = G.Apply(noise);
                    var d_logits = D.Apply(noise_z);
                    var d2_logits = D.Apply(imgs);

                    d_loss_real = BinaryCrossentropy(d2_logits, tf.ones_like(d2_logits));
                    d_loss_fake = BinaryCrossentropy(d_logits, tf.zeros_like(d_logits));

                    g_loss = BinaryCrossentropy(d_logits, tf.ones_like(d_logits));
                    d_loss = d_loss_real + d_loss_fake;
                    var grad = tape.gradient(d_loss, D.TrainableVariables);
                    d_optimizer.apply_gradients(zip(grad, D.TrainableVariables.Select(x => x as ResourceVariable)));

                    grad = tape.gradient(g_loss, G.TrainableVariables);
                    g_optimizer.apply_gradients(zip(grad, G.TrainableVariables.Select(x => x as ResourceVariable)));
                }
                if (i % 5 == 0 && i != 0)
                {
                    var s_d_loss_real = (float)tf.reduce_mean(d_loss_real).numpy();
                    var s_d_loss_fake = (float)tf.reduce_mean(d_loss_fake).numpy();
                    var s_d_loss = (float)tf.reduce_mean(d_loss).numpy();
                    var s_g_loss = (float)tf.reduce_mean(g_loss).numpy();
                    print($"step{i} d_loss:{s_d_loss}(Real: {s_d_loss_real} + Fake: {s_d_loss_fake}) g_loss:{s_g_loss}");
                    if (i % showstep == 0)
                        PredictImage(G, i);
                }
                if (i % 100 == 0)
                {
                    G.save_weights("dcgan\\models\\Model_" + i + "_g.weights");
                    D.save_weights("dcgan\\models\\Model_" + i + "_d.weights");
                }
            }
        }
    }

    private Tensor BinaryCrossentropy(Tensor x, Tensor y)
    {
        var shape = tf.reduce_prod(tf.shape(x));
        var count = tf.cast(shape, TF_DataType.TF_FLOAT);
        x = tf.clip_by_value(x, 1e-6f, 1.0f - 1e-6f);
        var z = y * tf.log(x) + (1 - y) * tf.log(1 - x);
        var result = -1.0f / count * tf.reduce_sum(z);
        return result;
    }

    private void PredictImage(Model g, int step)
    {
        var r = 5;
        var c = 5;

        var noise = np.random.normal(0, 1, new int[] { r * c, latent_dim });
        noise = noise.astype(np.float32);
        Tensor tensor_result = g.predict(noise);
        var gen_imgs = tensor_result.numpy();
        SaveImage(gen_imgs, step);
    }

    private void SaveImage(NDArray gen_imgs, int step)
    {
        int size = 4;
        gen_imgs = gen_imgs * 0.5 + 0.5; // 25x28x28x1 [0.0-1.0]
        //var c = 5;
        //var r = gen_imgs.shape[0] / c;
        //var nDArray = np.zeros((img_rows * r, img_cols * c), dtype: np.float32);
        //for (var i = 0; i < r; i++)
        //{
        //    for (var j = 0; j < c; j++)
        //    {
        //        var x = new Slice(i * img_rows, (i + 1) * img_cols);
        //        var y = new Slice(j * img_rows, (j + 1) * img_cols);
        //        var v = gen_imgs[i * r + j].reshape((img_rows, img_cols));
        //        nDArray[x, y] = v;
        //    }
        //}

        //var t = nDArray.reshape((img_rows * r, img_cols * c)) * 255;
        //GrayToRGB(t.astype(np.uint8)).ToBitmap().Save(imgpath + "/image" + step + ".jpg"); // .ToBitmap() missing

        var generatedImages = gen_imgs.ToArray(); // Tensor.Numpy[25]...

        Image image = new Bitmap(28 * 5 * size, 28 * 5 * size);
        for (int i = 0; i < generatedImages.Length; i++)
        {
            var values = generatedImages[i].reshape(784).ToArray<float>();
            float min = values.Min();
            float max = values.Max();
            float slope = 0.0f;
            if (max > min) slope = 255.0f / (max - min);
            Graphics g = Graphics.FromImage(image);
            for (int y = 0; y < 28; y++)
                for (int x = 0; x < 28; x++)
                {
                    var value = (int)((values[y * 28 + x] - min) * slope);
                    Brush thisBrush = new SolidBrush(Color.FromArgb(255, value, value, value));
                    g.FillRectangle(thisBrush, x * size + ((i % 5) * 28 * size), y * size + ((i / 5) * 28 * size), size, size);
                }
        }
        image.Save(imgpath + "/image" + (step / 10).ToString("D8") + ".jpg");



    }

    private NDArray GrayToRGB(NDArray img2D)
    {
        var img4A = np.full_like(img2D, 255);
        var img3D = np.expand_dims(img2D, 2);
        var r = np.dstack(img3D, img3D, img3D, img4A);
        var img4 = np.expand_dims(r, 0);
        return img4;
    }

    public override void Test()
    {
        var G = Make_Generator_model();
        G.load_weights(modelpath + "\\Model_100_g.weights");
        PredictImage(G, 1);
    }
}
