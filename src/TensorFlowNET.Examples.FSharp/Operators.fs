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

open NumSharp
open Tensorflow

[<AutoOpen>]
module TensorflowOperators =
    type ResourceVariable with
        member x.asTensor : Tensor = ResourceVariable.op_Implicit x

    type NDArray with
        member x.asTensor : Tensor = Tensor.op_Implicit x

    type Shape with
        member x.asTensorShape : TensorShape = TensorShape.op_Implicit x

    type Tensors with
        member x.asTensor : Tensor =
            match Seq.tryHead x with
            | Some t -> t
            | _ -> null

    type Tensor with
        member x.asTensors : Tensors = new Tensors([| x |])
