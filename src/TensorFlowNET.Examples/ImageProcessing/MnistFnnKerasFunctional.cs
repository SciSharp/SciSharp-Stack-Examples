/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using NumSharp;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// The Keras functional API is a way to create models that are more flexible than the tf.keras.Sequential API. 
    /// The functional API can handle models with non-linear topology, shared layers, and even multiple inputs or outputs.
    /// https://keras.io/guides/functional_api/
    /// </summary>
    public class MnistFnnKerasFunctional : SciSharpExample, IExample
    {
        Model model;
        NDArray x_train, y_train, x_test, y_test;

        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "MNIST FNN (Keras Functional)",
                Enabled = true,
                Priority = 17
            };

        public bool Run()
        {
            tf.enable_eager_execution();

            PrepareData();
            BuildModel();

            return true;
        }

        public override void PrepareData()
        {
            (x_train, y_train, x_test, y_test) = keras.datasets.mnist.load_data();
            x_train = x_train.reshape(60000, 784) / 255f;
            x_test = x_test.reshape(10000, 784) / 255f;
        }

        public override void BuildModel()
        {
            var inputs = keras.Input(shape: 784);

            var dense1 = layers.Dense(64, activation: tf.keras.activations.Relu);
            var outputs1 = dense1.Apply(inputs);

            var dense2 = layers.Dense(64, activation: tf.keras.activations.Relu);
            var outputs2 = dense2.Apply(outputs1);

            var dense3 = layers.Dense(10);
            var outputs3 = dense3.Apply(outputs2);

            model = keras.Model(inputs, outputs3, name: "mnist_model");
            model.summary();

            model.compile(loss: keras.losses.SparseCategoricalCrossentropy(from_logits: true),
                optimizer: keras.optimizers.RMSprop(),
                metrics: new[] { "accuracy" });

            model.fit(x_train, y_train, batch_size: 64, epochs: 2, validation_split: 0.2f);
        }
    }
}
