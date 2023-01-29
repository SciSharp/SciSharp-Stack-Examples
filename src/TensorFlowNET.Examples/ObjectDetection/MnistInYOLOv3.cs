/*****************************************************************************
   Copyright 2021 Haiping Chen. All Rights Reserved.

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
using SciSharp.Models.ObjectDetection;
using System;
using System.IO;

namespace TensorFlowNET.Examples;

public class MnistInYOLOv3 : SciSharpExample, IExample
{
    YoloConfig cfg;
    float accuracy_test = 0f;
    YoloDataset trainingData, testingData;
    
    public ExampleConfig InitConfig()
        => Config = new ExampleConfig
        {
            Name = "MNIST in YOLOv3",
            Enabled = false
        };

    public bool Run()
    {
        cfg = new YoloConfig("YOLOv3");
        (trainingData, testingData) = PrepareData();
        Train();
        Test();
        return true;
    }

    public override void Train()
    {
        // using wizard to train model
        var wizard = new ModelWizard();
        var task = wizard.AddObjectDetectionTask<YOLOv3>(new TaskOptions
        {
            InputShape = (28, 28, 1),
            NumberOfClass = 10,
        });
        task.SetModelArgs(cfg);

        task.Train(new YoloTrainingOptions
        {
            TrainingData = trainingData,
            TestingData = testingData
        });
    }

    public override void Test()
    {
        var wizard = new ModelWizard();
        var task = wizard.AddObjectDetectionTask<YOLOv3>(new TaskOptions
        {
            ModelPath = @"./YOLOv3/yolov3.h5"
        });
        task.SetModelArgs(cfg);
        var result = task.Test(new TestingOptions
        {
            
        });
        accuracy_test = result.Accuracy;
    }

    public (YoloDataset, YoloDataset) PrepareData()
    {
        string dataDir = Path.Combine("YOLOv3", "data");
        Directory.CreateDirectory(dataDir);

        var trainset = new YoloDataset("train", cfg);
        var testset = new YoloDataset("test", cfg);
        return (trainset, testset);
    }
}
