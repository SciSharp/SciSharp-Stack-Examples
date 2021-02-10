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

open System
open System.Diagnostics
open System.Drawing
open System.Reflection

open Argu
open Tensorflow
open type Tensorflow.Binding
open type Tensorflow.KerasApi

open TensorFlowNET.Examples.FSharp

let printc (color : Color) (s : string) = Colorful.Console.WriteLine(s, color)

let printEnv () =
    $"{Environment.OSVersion}\n\
      64Bit Operating System: {Environment.Is64BitOperatingSystem}\n\
      .NET CLR: {Environment.Version}\n\
      TensorFlow Binary v{tf.VERSION}\n\
      TensorFlow.NET v{Assembly.GetAssembly(typeof<TF_DataType>).GetName().Version}\n\
      TensorFlow.Keras v{Assembly.GetAssembly(typeof<KerasApi>).GetName().Version}\n\
      {Environment.CurrentDirectory}" |> printc Color.Yellow

let runExamples choice examples =
    let sw = Stopwatch()

    let runExample (example : SciSharpExample) =
        printc Color.White $"{DateTime.UtcNow} Starting {example.Config.Name}"

        let result =
            try
                sw.Restart()
                let isSuccess = example.Run()
                sw.Stop()

                $"Example: %s{example.Config.Name} in %f{sw.Elapsed.TotalSeconds}s"
                |> if isSuccess then Ok else Error
            with
            | ex ->
                printfn "%A" ex
                Error $"Example: %s{example.Config.Name}"

        keras.backend.clear_session()
    
        printc Color.White $"{DateTime.UtcNow} Completed {example.Config.Name}"
        result

    let isSelected id =
        match choice with
        | Some x -> id = x
        | None -> true

    let results =
        examples
        |> List.mapi (fun i example -> i + 1, example)
        |> List.where (fst >> isSelected)
        |> List.map (snd >> runExample)

    results |> List.iter (function
        | Ok s -> printc Color.Green $"{s} is OK!"
        | Error s -> printc Color.Red $"{s} is Failed!")

    $"TensorFlow Binary v{tf.VERSION}\n\
      TensorFlow.NET v{Assembly.GetAssembly(typeof<TF_DataType>).GetName().Version}\n\
      TensorFlow.Keras v{Assembly.GetAssembly(typeof<KerasApi>).GetName().Version}\n\
      {results.Length} of {examples.Length} example(s) are completed.\n\
      Press [Enter] to continue..." |> printfn "%s"
    Console.ReadLine().Length = 0

type Arguments =
    | [<AltCommandLine("-ex")>] Example of name : string

    interface IArgParserTemplate with
        member s.Usage =
            match s with
            | Example _ -> "name of selected example."

[<EntryPoint>]
let main argv =
    let parser = ArgumentParser.Create<Arguments>(programName = "TensorFlowNET.Examples.FSharp.exe")
    let arguments =
        try
            let results = parser.Parse argv
            results.GetAllResults() |> Some
        with
        | _ ->
            parser.PrintUsage() |> printc Color.Red
            Option.None

    match arguments with
    | None -> 1

    | Some args ->
        
        printEnv()

        let isSelected (example: SciSharpExample) =
            args.Length = 0 || List.contains (Example example.Config.Name) args

        let examples =
            //FunctionApproximation.run() // Still needs updates
            [ HelloWorld.Example
              BasicOperations.Example
              LinearRegression.Example
              LinearRegressionEager.Example
              LogisticRegression.Example
              LogisticRegressionEager.Example
              NearestNeighbor.Example
              NaiveBayesClassifier.Example
              FullyConnectedEager.Example
              FullyConnectedKeras.Example
            ]
            |> List.where isSelected
            |> List.sortBy (fun e -> e.Config.Priority)

        if examples.Length = 0 then
            printc Color.Red "No example found!"
            1
        else
            let (|ExampleId|_|) str =
                match Int32.TryParse(str : string) with
                | (true,id) ->
                    if id >= 1 && id <= examples.Length then Some(id) else Option.None
                | _ -> Option.None

            let rec loop () =
                examples |> List.iteri (fun i e -> printfn $"[{i + 1}]: {e.Config.Name}")

                let choice =
                    if examples.Length = 1
                    then Some 1
                    else
                        printc Color.Yellow $"Choose one example to run, hit [Enter] to run all: "
                        match Console.ReadLine() with
                        | ExampleId id -> Some id
                        | _ -> Option.None

                if runExamples choice examples
                then loop()
                else 0

            loop()
