using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorFlowNET.Examples
{
    public class WeatherPrediction : SciSharpExample, IExample
    {
        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "Weather Prediction",
                Enabled = true
            };

        public bool Run()
        {
            return true;
        }
    }
}
