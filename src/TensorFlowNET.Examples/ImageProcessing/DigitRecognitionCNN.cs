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

using SciSharp.Models;
using SciSharp.Models.ImageClassification;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

namespace TensorFlowNET.Examples;

/// <summary>
/// Convolutional Neural Network classifier for Hand Written Digits
/// CNN architecture with two convolutional layers, followed by two fully-connected layers at the end.
/// Use Stochastic Gradient Descent (SGD) optimizer. 
/// https://www.easy-tensorflow.com/tf-tutorials/convolutional-neural-nets-cnns/cnn1
/// </summary>
public class DigitRecognitionCNN : SciSharpExample, IExample
{
    Datasets<MnistDataSet> mnist;

    float accuracy_test = 0f;

    NDArray x_train, y_train;
    NDArray x_valid, y_valid;
    NDArray x_test, y_test;

    public ExampleConfig InitConfig()
        => Config = new ExampleConfig
        {
            Name = "MNIST CNN (Graph)",
            Enabled = true
        };

    public bool Run()
    {
        PrepareData();
        Train();
        Test();
        Predict();

        return accuracy_test > 0.95;
    }

    public override void Train()
    {
        // using wizard to train model
        var wizard = new ModelWizard();
        var task = wizard.AddImageClassificationTask<CNN>(new TaskOptions
        {
            InputShape = (28, 28, 1),
            NumberOfClass = 10,
        });
        task.SetModelArgs(new ConvArgs
        {
            NumberOfNeurons = 128
        });
        task.Train(new TrainingOptions
        {
            Epochs = 5,
            TrainingData = new FeatureAndLabel(x_train, y_train),
            ValidationData = new FeatureAndLabel(x_valid, y_valid)
        });
    }

    public override void Test()
    {
        var wizard = new ModelWizard();
        var task = wizard.AddImageClassificationTask<CNN>(new TaskOptions
        {
            ModelPath = @"image_classification_cnn_v1\saved_model.pb"
        });
        var result = task.Test(new TestingOptions
        {
            TestingData = new FeatureAndLabel(x_test, y_test)
        });
        accuracy_test = result.Accuracy;
    }

    public override void Predict()
    {
        // predict image
        var wizard = new ModelWizard();
        var task = wizard.AddImageClassificationTask<CNN>(new TaskOptions
        {
            LabelPath = @"image_classification_cnn_v1\labels.txt",
            ModelPath = @"image_classification_cnn_v1\saved_model.pb"
        });

        var input = x_test["0:1"];
        var result = task.Predict(input);
        long output = np.argmax(y_test[0]);
        Debug.Assert(result.Label == output.ToString());

        input = x_test["1:2"];
        result = task.Predict(input);
        output = np.argmax(y_test[1]);
        Debug.Assert(result.Label == output.ToString());
    }

    public override void PrepareData()
    {
        Directory.CreateDirectory("image_classification_cnn_v1");
        var loader = new MnistModelLoader();
        mnist = loader.LoadAsync(".resources/mnist", oneHot: true, showProgressInConsole: true).Result;
        (x_train, y_train) = Reformat(mnist.Train.Data, mnist.Train.Labels);
        (x_valid, y_valid) = Reformat(mnist.Validation.Data, mnist.Validation.Labels);
        (x_test, y_test) = Reformat(mnist.Test.Data, mnist.Test.Labels);

        print("Size of:");
        print($"- Training-set:\t\t{len(mnist.Train.Data)}");
        print($"- Validation-set:\t{len(mnist.Validation.Data)}");

        // generate labels
        var labels = range(0, 10).Select(x => x.ToString());
        File.WriteAllLines(@"image_classification_cnn_v1\labels.txt", labels);
    }

    /// <summary>
    /// Reformats the data to the format acceptable for convolutional layers
    /// </summary>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    private (NDArray, NDArray) Reformat(NDArray x, NDArray y)
    {
        var (unique_y, _) = np.unique(np.argmax(y, 1));
        var (img_size, num_ch, num_class) = ((int)np.sqrt(x.shape[1]).astype(np.int32), 1, len(unique_y));
        var dataset = x.reshape((x.shape[0], img_size, img_size, num_ch)).astype(np.float32);
        //y[0] = np.arange(num_class) == y[0];
        //var labels = (np.arange(num_class) == y.reshape(y.shape[0], 1, y.shape[1])).astype(np.float32);
        return (dataset, y);
    }
}
