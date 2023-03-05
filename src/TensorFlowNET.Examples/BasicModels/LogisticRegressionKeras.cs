/*****************************************************************************
   Copyright 2023 Haiping Chen. All Rights Reserved.

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

using System.Collections.Generic;
using Tensorflow.Keras;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Examples
{
    public class LogisticRegressionKeras : SciSharpExample, IExample
    {
        ICallback result;

        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "Logistic Regression (Keras)",
                Enabled = true,
                IsImportingGraph = false
            };

        public bool Run()
        {
            tf.enable_eager_execution();

            // Prepare MNIST data.
            var ((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data();
            
            // Normalize images value from [0, 255] to [0, 1].
            (x_train, x_test) = (x_train / 255f, x_test / 255f);

            var model = keras.Sequential(new List<ILayer>
            {
                // Flatten images to 1-D vector of 784 features (28*28).
                keras.layers.Flatten(),
                keras.layers.Dense(10, activation: "softmax")
            });

            // Compile the model, specifying that SGD should be used to train and the cross entropy 
            // loss function should be used. Also keep track of accuracy throughout training.
            model.compile(optimizer: keras.optimizers.SGD(0.01f),
                loss: keras.losses.SparseCategoricalCrossentropy(from_logits: true),
                metrics: new[] { "accuracy" });

            result = model.fit(x_train, y_train, epochs: 5);

            // model.evaluate(x_test, np.argmax(y_test));

            var predicted = model.predict(x_test);

            return true;
        }
    }
}
