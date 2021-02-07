﻿(******************************************************************************
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
******************************************************************************)

namespace TensorFlowNET.Examples.FSharp

open NumSharp
open Tensorflow
open type Tensorflow.Binding
open type Tensorflow.KerasApi

/// <summary>
/// A linear regression learning algorithm example using TensorFlow library.
/// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/2_BasicModels/linear_regression.ipynb
/// </summary>

module LinearRegressionEager =
    let private run() =
        let tf = New<tensorflow>()
        tf.enable_eager_execution()
        
        let training_steps = 1000
        let learning_rate = 0.01f
        let display_step = 100
        
        let train_X = 
            np.array(3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 7.59f, 2.167f,
                     7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f)
        let train_Y = 
            np.array(1.7f, 2.76f, 2.09f, 3.19f, 1.694f, 1.573f, 3.366f, 2.596f, 2.53f, 1.221f,
                     2.827f, 3.465f, 1.65f, 2.904f, 2.42f, 2.94f, 1.3f)
        let n_samples = train_X.shape.[0]
        
        // We can set a fixed init value in order to demo
        let W = tf.Variable(-0.06f,name = "weight")
        let b = tf.Variable(-0.73f, name = "bias")
        let optimizer = keras.optimizers.SGD(learning_rate)
        
        // Run training for the given number of steps.
        for step = 1 to (training_steps + 1) do 
            // Run the optimization to update W and b values.
            // Wrap computation inside a GradientTape for automatic differentiation.
            use g = tf.GradientTape()
            // Linear regression (Wx + b).
            let pred = W * train_X + b
            // Mean square error.
            let loss = tf.reduce_sum(tf.pow(pred - train_Y,2)) / (2 * n_samples)
            // should stop recording
            // compute gradients
            let gradients = g.gradient(loss,struct (W,b))
        
            // Update W and b following gradients.
            optimizer.apply_gradients(zip(gradients, struct (W,b)))
        
            if (step % display_step) = 0 then
                let pred = W * train_X + b
                let loss = tf.reduce_sum(tf.pow(pred-train_Y,2)) / (2 * n_samples)
                printfn $"step: {step}, loss: {loss.numpy()}, W: {W.numpy()}, b: {b.numpy()}"

        true
    
    let Example =
        { SciSharpExample.Config = ExampleConfig.Create ("Linear Regression (Eager)", priority0 = 5)
          Run = run
        }
