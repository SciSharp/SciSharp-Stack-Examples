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

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Reflection;
using Tensorflow;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Console = Colorful.Console;

namespace TensorFlowNET.Examples
{
    class Program
    {
        static void Main(string[] args)
        {
            var parsedArgs = ParseArgs(args);

            var examples = Assembly.GetEntryAssembly().GetTypes()
                .Where(x => x.GetInterfaces().Contains(typeof(IExample)))
                //.Where(x => x.Name == nameof(MnistGAN))
                //.Where(x => x.Name == nameof(ImageClassificationKeras))
                //.Where(x => x.Name == nameof(WeatherPrediction))
                .ToArray();

            Console.WriteLine(Environment.OSVersion, Color.Yellow);
            Console.WriteLine($"64Bit Operating System: {Environment.Is64BitOperatingSystem}", Color.Yellow);
            Console.WriteLine($".NET CLR: {Environment.Version}", Color.Yellow);
            Console.WriteLine($"TensorFlow Binary v{tf.VERSION}", Color.Yellow);
            Console.WriteLine($"TensorFlow.NET v{Assembly.GetAssembly(typeof(TF_DataType)).GetName().Version}", Color.Yellow);
            Console.WriteLine($"TensorFlow.Keras v{Assembly.GetAssembly(typeof(KerasApi)).GetName().Version}", Color.Yellow);
            Console.WriteLine(Environment.CurrentDirectory, Color.Yellow);

            int finished = 0;
            var errors = new List<string>();
            var success = new List<string>();

            var sw = new Stopwatch();

            for (var i = 0; i < examples.Length; i++)
            {
                var (isSuccess, name) = (true, "");
                sw.Restart();
                (isSuccess, name) = RunExamples(examples[i], parsedArgs);
                sw.Stop();

                if (isSuccess)
                    success.Add($"Example: {name} in {sw.Elapsed.TotalSeconds}s");
                else
                    errors.Add($"Example: {name} in {sw.Elapsed.TotalSeconds}s");

                finished++;
                keras.backend.clear_session();
            }

            success.ForEach(x => Console.WriteLine($"{x} is OK!", Color.White));
            errors.ForEach(x => Console.WriteLine($"{x} is Failed!", Color.Red));

            Console.WriteLine(Environment.OSVersion, Color.Yellow);
            Console.WriteLine($"64Bit Operating System: {Environment.Is64BitOperatingSystem}", Color.Yellow);
            Console.WriteLine($".NET CLR: {Environment.Version}", Color.Yellow);
            Console.WriteLine($"TensorFlow Binary v{tf.VERSION}");
            Console.WriteLine($"TensorFlow.NET v{Assembly.GetAssembly(typeof(TF_DataType)).GetName().Version}");
            Console.WriteLine($"TensorFlow.Keras v{Assembly.GetAssembly(typeof(KerasApi)).GetName().Version}");
            Console.WriteLine($"{finished} of {examples.Length} example(s) are completed.");
            Console.ReadLine();
        }

        private static (bool, string) RunExamples(Type example, Dictionary<string, string> args)
        {
            var instance = (IExample)Activator.CreateInstance(example);
            instance.InitConfig();
            var name = instance.Config.Name;

            Console.WriteLine($"{DateTime.UtcNow} Starting {name}", Color.White);

            // if (args.ContainsKey("ex") && name != args["ex"])
                // return (false, "");

            if (!instance.Config.Enabled)
                return (true, name);

            bool ret = false;
            try
            {
                ret = instance.Run();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex);
            }
            
            Console.WriteLine($"{DateTime.UtcNow} Completed {name}", Color.White);
            return (ret, name);
        }

        private static Dictionary<string, string> ParseArgs(string[] args)
        {
            var parsed = new Dictionary<string, string>();

            for (int i = 0; i < args.Length; i++)
            {
                string key = args[i].Substring(1);
                switch (key)
                {
                    case "ex":
                        parsed.Add(key, args[++i]);
                        break;
                    default:
                        break;
                }
            }

            return parsed;
        }
    }
}
