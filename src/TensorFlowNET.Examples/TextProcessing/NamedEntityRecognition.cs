using System;
using Tensorflow;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// https://github.com/guillaumegenthial/tf_ner
    /// </summary>
    public class NamedEntityRecognition : SciSharpExample, IExample
    {
        public ExampleConfig InitConfig()
            => Config = new ExampleConfig
            {
                Name = "NER",
                Enabled = false,
                IsImportingGraph = false
            };

        public bool Run()
        {
            throw new NotImplementedException();
        }
    }
}
