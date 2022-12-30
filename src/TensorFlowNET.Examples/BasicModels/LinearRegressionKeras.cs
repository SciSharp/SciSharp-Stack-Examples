/*****************************************************************************
   Copyright 2020 Haiping Chen. All Rights Reserved.

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

using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Examples
{
    public class LinearRegressionKeras : SciSharpExample, IExample
    {
        NDArray train_X, train_Y;

        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "Linear Regression (Keras)",
                Enabled = true,
                IsImportingGraph = false
            };

        public bool Run()
        {
            tf.enable_eager_execution();

            PrepareData();

            BuildModel();

            return true;
        }

        public override void BuildModel()
        {
            var inputs = keras.Input(shape: 1);
            var outputs = layers.Dense(1).Apply(inputs);
            var model = keras.Model(inputs, outputs);

            model.summary();
            model.compile(loss: keras.losses.MeanSquaredError(),
                optimizer: keras.optimizers.SGD(0.005f),
                metrics: new[] { "acc" });
            model.fit(train_X, train_Y, epochs: 100);

            var weights = model.TrainableVariables;
            print($"weight: {weights[0].numpy()}, bias: {weights[1].numpy()}");
        }

        public override void PrepareData()
        {
            train_X = np.array(3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 
                    7.59f, 2.167f, 7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f);

            train_Y = np.array(1.7f, 2.76f, 2.09f, 3.19f, 1.694f, 1.573f, 3.366f, 
                    2.596f, 2.53f, 1.221f, 2.827f, 3.465f, 1.65f, 2.904f, 2.42f, 2.94f, 1.3f);
        }
    }
}
