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
using Tensorflow;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Utils;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// A toy ResNet model
    /// https://keras.io/guides/functional_api/
    /// </summary>
    public class CIFAR10_CNN : SciSharpExample, IExample
    {
        Model model;
        NDArray x_train, y_train;
        NDArray x_test, y_test;

        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "Toy ResNet",
                Enabled = true
            };

        public bool Run()
        {
            tf.enable_eager_execution();
            
            BuildModel();
            PrepareData();
            Train();

            return true;
        }

        public override void BuildModel()
        {
            var inputs = keras.Input(shape: (32, 32, 3), name: "img");
            var x = layers.Conv2D(32, 3, activation: "relu").Apply(inputs);
            x = layers.Conv2D(64, 3, activation: "relu").Apply(x);

            // x = layers.BatchNormalization().Apply(x);
            var block_1_output = layers.MaxPooling2D(3).Apply(x);

            x = layers.Conv2D(64, 3, activation: "relu", padding: "same").Apply(block_1_output);
            x = layers.Conv2D(64, 3, activation: "relu", padding: "same").Apply(x);
            var block_2_output = layers.Add().Apply(new Tensors(x, block_1_output));

            x = layers.Conv2D(64, 3, activation: "relu", padding: "same").Apply(block_2_output);
            x = layers.Conv2D(64, 3, activation: "relu", padding: "same").Apply(x);
            var block_3_output = layers.Add().Apply(new Tensors(x, block_2_output));

            x = layers.Conv2D(64, 3, activation: "relu").Apply(block_3_output);
            x = layers.GlobalAveragePooling2D().Apply(x);
            x = layers.Dense(256, activation: "relu").Apply(x);
            x = layers.Dropout(0.5f).Apply(x);
            var outputs = layers.Dense(10).Apply(x);

            model = keras.Model(inputs, outputs, name: "toy_resnet");
            model.summary();

            model.compile(
                optimizer: keras.optimizers.RMSprop(1e-3f),
                loss: keras.losses.CategoricalCrossentropy(from_logits: true),
                metrics: new[] { "acc" });
        }

        public override void PrepareData()
        {
            ((x_train, y_train), (x_test, y_test)) = keras.datasets.cifar10.load_data();

            x_train = x_train / 255.0f;
            x_test = x_test / 255.0f;

            y_train = np_utils.to_categorical(y_train, 10);
            y_test = np_utils.to_categorical(y_test, 10);
        }

        public override void Train()
        {
            model.fit(x_train[new Slice(0, 2000)], y_train[new Slice(0, 2000)], 
                batch_size: 64, 
                epochs: 3, 
                validation_split: 0.2f);
        }
    }
}
