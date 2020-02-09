using System;
using System.Collections.Generic;
using System.Text;

namespace TensorFlowNET.Examples
{
    public class ExampleConfig
    {
        public int Priority { get; set; } = 100;

        /// <summary>
        /// Example name
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// True to run example
        /// </summary>
        public bool Enabled { get; set; }

        /// <summary>
        /// Set true to import the computation graph instead of building it.
        /// </summary>
        public bool IsImportingGraph { get; set; }
    }
}
