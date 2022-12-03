using System.Diagnostics;
using Tensorflow;
using Tensorflow.Keras.Engine;
using Tensorflow.NumPy;
using static Tensorflow.KerasApi;

namespace WebApi.Services
{
    public class ImageClassificationService
    {
        const float learningRate = 0.001f;
        const int batchSize = 100;
        const int epochs = 10;

        private Sequential baseModel;
        private Sequential predictionModel;

        private NDArray trainImages;
        private NDArray trainLabels;

        private NDArray testImages;
        private NDArray testLabels;

        public ImageClassificationService()
        {

        }

        public void Train()
        {
            // !!
            // Here to silence CS8618 warning
            trainImages = default!;
            trainLabels = default!;
            testImages = default!;
            testLabels = default!;
            baseModel = default!;
            // !!

            // Prepare datasets
            PrepareData();

            // Prepare the model
            PrepareModel();

            // Train the model
            {
                var stopwatch = new Stopwatch();
                Console.WriteLine("Starting training...");
                stopwatch.Start();

                baseModel.fit(trainImages, trainLabels, batchSize, epochs);

                stopwatch.Stop();
                Console.WriteLine($"Took {stopwatch.ElapsedMilliseconds / 1000} seconds");
            }

            baseModel.evaluate(testImages, testLabels);
        }


        private void PrepareData()
        {
            // Load the MNIST dataset
            var data = keras.datasets.mnist.load_data();
            (trainImages, trainLabels) = data.Train;
            (testImages, testLabels) = data.Test;

            // Pre-process it, turning them all into doubles between 0-1 instead of bytes between 0-255. We do this even
            // though we receive data from the user as raw bytes (0-255), neural networks work better with doubles than
            // integers
            trainImages /= 255.0f;
            testImages /= 255.0f;
        }


        private void PrepareModel()
        {
            baseModel = keras.Sequential();
            baseModel.add(keras.layers.Input((28, 28)));
            baseModel.add(keras.layers.Flatten());
            baseModel.add(keras.layers.Dense(256, activation: "relu"));
            baseModel.add(keras.layers.Dense(128, activation: "relu"));
            baseModel.add(keras.layers.Dense(10));

            baseModel.compile(
                loss: keras.losses.SparseCategoricalCrossentropy(from_logits: true),
                optimizer: keras.optimizers.Adam(learningRate),
                metrics: new string[] { "accuracy" }
            );
        }

        public float[] Predict(byte[] data)
        {
            if (data.Length != 28 * 28)
            {
                Console.Error.WriteLine("Data must be exactly 28x28 (784) bytes. Truncating or padding to match.");
                Array.Resize(ref data, 28 * 28);
            }

            // Reshape our input data into 1x (28x28) arrays (AKA one 1x28x28 tensor, AKA a 3D vector)
            var array = np.array(data);
            array = array.reshape((1, 28, 28));
            array = array / 255.0f;

            // Create a copy of the same model, but with a Softmax layer at the end to get outputs.
            if(predictionModel == null)
            {
                predictionModel = keras.Sequential();
                predictionModel.add(baseModel);
                predictionModel.add(keras.layers.Softmax(-1));
                predictionModel.compile();
            }

            Tensors predictions = predictionModel.predict(array);

            var prediction = predictions[0].numpy().reshape(-1);
            return prediction.ToArray<float>();
        }
    }
}
