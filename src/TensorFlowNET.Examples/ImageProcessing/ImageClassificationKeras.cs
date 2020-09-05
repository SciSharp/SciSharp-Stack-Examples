using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// This tutorial shows how to classify images of flowers.
    /// https://www.tensorflow.org/tutorials/images/classification
    /// </summary>
    public class ImageClassificationKeras : SciSharpExample, IExample
    {
        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "Image Classification (Keras)",
                Enabled = true,
                Priority = 18
            };

        public bool Run()
        {
            PrepareData();
            return true;
        }

        public override void PrepareData()
        {
        }
    }
}
