(*****************************************************************************
Copyright 2021 The TensorFlow.NET Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
******************************************************************************)

namespace TensorFlowNET.Examples.FSharp

type ExampleConfig =
    { Name : string
      Priority : int
      /// True to run example
      Enabled : bool
      /// Set true to import the computation graph instead of building it
      IsImportedGraph : bool
    }
    static member Create (name, ?priority0, ?enabled0, ?isImportedGraph0) =
        let priority = defaultArg priority0 100
        let enabled = defaultArg enabled0 true
        let isImportedGraph = defaultArg isImportedGraph0 false
        { Name = name
          Priority = priority
          Enabled = enabled
          IsImportedGraph = isImportedGraph
        }

type SciSharpExample =
    { Config : ExampleConfig
      Run : unit -> bool
    }
