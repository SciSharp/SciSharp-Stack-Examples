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
using System;
using System.Collections.Generic;
using System.IO;
using Tensorflow.Keras.Utils;

namespace TensorFlowNET.Examples;

/// <summary>
/// In this tutorial, we will reuse the feature extraction capabilities from powerful image classifiers trained on ImageNet 
/// and simply train a new classification layer on top. Transfer learning is a technique that shortcuts much of this 
/// by taking a piece of a model that has already been trained on a related task and reusing it in a new model.
/// 
/// https://www.tensorflow.org/hub/tutorials/image_retraining
/// </summary>
public class TransferLearningWithInceptionV3 : SciSharpExample, IExample
{
    float accuracy;
    public ExampleConfig InitConfig()
        => Config = new ExampleConfig
        {
            Name = "Transfer Learning With InceptionV3 (Graph)",
            Enabled = true
        };

    public bool Run()
    {
        PrepareData();
        Train();
        Test();
        Predict();
        
        return accuracy > 0.75f;
    }

    public override void PrepareData()
    {
        // get a set of images to teach the network about the new classes
        string fileName = "flower_photos.tgz";
        string dataDir = "image_classification_v1";
        string url = $"http://download.tensorflow.org/example_images/{fileName}";
        Web.Download(url, dataDir, fileName);
        Compress.ExtractTGZ(Path.Join(dataDir, fileName), dataDir);
    }

    public override void Train()
    {
        // using wizard to train model
        var wizard = new ModelWizard();
        var task = wizard.AddImageClassificationTask<TransferLearning>(new TaskOptions
        {
            DataDir = @"image_classification_v1\flower_photos",
        });
        task.Train(new TrainingOptions
        {
            TrainingSteps = 100
        });
    }

    /// <summary>
    /// Prediction
    /// labels mapping, it's from output_lables.txt
    /// 0 - daisy
    /// 1 - dandelion
    /// 2 - roses
    /// 3 - sunflowers
    /// 4 - tulips
    /// </summary>
    public override void Predict()
    {
        // predict image
        var wizard = new ModelWizard();
        var task = wizard.AddImageClassificationTask<TransferLearning>(new TaskOptions
        {
            ModelPath = @"image_classification_v1\saved_model.pb"
        });

        var imgPath = Path.Join("image_classification_v1", "flower_photos", "daisy", "5547758_eea9edfd54_n.jpg");
        var input = ImageUtil.ReadImageFromFile(imgPath);
        var result = task.Predict(input);
    }

    public override void Test()
    {
        var wizard = new ModelWizard();
        var task = wizard.AddImageClassificationTask<TransferLearning>(new TaskOptions
        {
            DataDir = @"image_classification_v1\flower_photos",
            ModelPath = @"image_classification_v1\saved_model.pb"
        });
        var result = task.Test(new TestingOptions
        {
        });
        accuracy = result.Accuracy;
    }
}
