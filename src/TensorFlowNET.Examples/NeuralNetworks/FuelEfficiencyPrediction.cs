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

using static Tensorflow.Binding;
using static PandasNet.PandasApi;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Examples;

/// <summary>
/// This tutorial uses the classic Auto MPG dataset and demonstrates how to build models to 
/// predict the fuel efficiency of the late-1970s and early 1980s automobiles.
/// https://www.tensorflow.org/tutorials/keras/regression
/// </summary>
public class FuelEfficiencyPrediction : SciSharpExample, IExample
{
    public ExampleConfig InitConfig()
        => Config = new ExampleConfig
        {
            Name = "Predict fuel efficiency",
            Enabled = true
        };

    public bool Run()
    {
        PrepareData();
        return true;
    }

    public override void PrepareData()
    {
        string url = $"http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data";
        var dataset = pd.read_csv(url, 
            names: new[] 
            {
                "MPG", "Cylinders", "Displacement", "Horsepower", 
                "Weight", "Acceleration", "Model Year", "Origin" 
            },
            sep: ' ',
            na_values: '?',
            comment: '\t',
            skipinitialspace: true);
        dataset = dataset.dropna();

        // The "Origin" column is categorical, not numeric.
        // So the next step is to one-hot encode the values in the column with pd.get_dummies.
        dataset["Origin"] = dataset["Origin"].map<string, string>((i) => i switch
        {
            "1" => "USA",
            "2" => "Europe",
            "3" => "Japan",
            _ => "N/A"
        });

        dataset = pd.get_dummies(dataset, columns: new[] { "Origin" }, prefix: "", prefix_sep: "");

        var train_dataset = dataset.sample(frac: 0.8f, random_state: 0);
        var test_dataset = dataset.drop(train_dataset.index.array<int>());

        var train_features = train_dataset.copy();
        var test_features = test_dataset.copy();

        var train_labels = train_features.pop("MPG");
        var test_labels = test_features.pop("MPG");

        // var df = train_dataset.describe().transpose()["mean", "std"];

        var normalizer = tf.keras.layers.Normalization(axis: -1);
        normalizer.adapt(train_features);

        // Linear regression
        var horsepower = train_features["Horsepower"];

        var horsepower_normalizer = layers.Normalization(input_shape: 1, axis: null);
        horsepower_normalizer.adapt(horsepower);

        var horsepower_model = keras.Sequential(horsepower_normalizer,
            layers.Dense(units: 1));

        horsepower_model.summary();

        horsepower_model.compile(
            optimizer: tf.keras.optimizers.Adam(learning_rate: 0.1f),
            loss: tf.keras.losses.MeanAbsoluteError());

        var history = horsepower_model.fit(
            train_features["Horsepower"],
            train_labels,
            epochs: 100,
            // Suppress logging.
            verbose: 1,
            // Calculate validation results on 20% of the training data.
            validation_split: 0.2f);

        var results = horsepower_model.evaluate(
            test_features["Horsepower"],
            test_labels, verbose: 1);

        // Linear regression with multiple inputs
        var linear_model = keras.Sequential(normalizer,
            layers.Dense(units: 1));

        linear_model.compile(
            optimizer: tf.keras.optimizers.Adam(learning_rate: 0.1f),
            loss: tf.keras.losses.MeanAbsoluteError());

        history = linear_model.fit(
            train_features,
            train_labels,
            epochs: 100,
            verbose: 1,
            validation_split: 0.2f);

        linear_model.evaluate(
            test_features, test_labels, verbose: 1);

        // Regression with a deep neural network (DNN)
        var dnn_model = keras.Sequential(normalizer,
            layers.Dense(64, activation: "relu"),
            layers.Dense(64, activation: "relu"),
            layers.Dense(1));

        dnn_model.compile(
            optimizer: tf.keras.optimizers.Adam(learning_rate: 0.001f),
            loss: tf.keras.losses.MeanAbsoluteError());

        history = dnn_model.fit(
            train_features,
            train_labels,
            epochs: 100,
            verbose: 1,
            validation_split: 0.2f);

        dnn_model.evaluate(
            test_features, test_labels, verbose: 1);
    }
}
