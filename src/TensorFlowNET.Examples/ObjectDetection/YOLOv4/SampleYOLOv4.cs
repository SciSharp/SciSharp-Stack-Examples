using System;
using System.Collections.Generic;
using System.Text;

namespace TensorFlowNET.Examples.ObjectDetection.YOLOv4
{
    /// <summary>
    /// https://github.com/hunglc007/tensorflow-yolov4-tflite
    /// </summary>
    public class SampleYOLOv4 : SciSharpExample, IExample
    {
        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "YOLOv4 (Keras)",
                Enabled = false
            };

        public bool Run()
        {
            return true;
        }
    }
}
