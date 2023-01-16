using SciSharp.Models.ImageClassification;
using SciSharp.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlowNET.Examples;
using SciSharp.Models.TextClassification;

namespace TensorFlowNET.NaturalLanguageProcessing
{
    public class TextGeneration : SciSharpExample, IExample
    {
        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "Text Generation",
                Enabled = true
            };

        public bool Run()
        {
            PrepareData();
            Train();
            return true;
        }

        public override void PrepareData()
        {
            
        }

        public override void Train()
        {
            // using wizard to train model
            var wizard = new ModelWizard();
            var task = wizard.AddTextGenerationTask<RnnTextGenerator>(new TaskOptions
            {
                DataDir = @"",
            });
            task.Train(new TrainingOptions
            {
                TrainingSteps = 100
            });
        }
    }
}
