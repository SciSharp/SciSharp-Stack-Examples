// Learn more about F# at http://fsharp.org

open System
open TensorFlow.NET.Examples.FSharp

[<EntryPoint>]
let main argv =
    //FunctionApproximation.run() // Still Needs updates
    LinearRegressionEager.run()
    0 // return an integer exit code
