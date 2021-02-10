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

open System

open NumSharp
open System.IO
open Tensorflow
open Tensorflow.Keras.Utils
open type Tensorflow.Binding

/// https://github.com/nicolov/naive_bayes_tensorflow
module NaiveBayesClassifier =

    let private prepareData () =

        let X = np.array(array2D [
            [5.1f; 3.5f]; [4.9f; 3.0f]; [4.7f; 3.2f]; [4.6f; 3.1f]; [5.0f; 3.6f]; [5.4f; 3.9f];
            [4.6f; 3.4f]; [5.0f; 3.4f]; [4.4f; 2.9f]; [4.9f; 3.1f]; [5.4f; 3.7f]; [4.8f; 3.4f];
            [4.8f; 3.0f]; [4.3f; 3.0f]; [5.8f; 4.0f]; [5.7f; 4.4f]; [5.4f; 3.9f]; [5.1f; 3.5f];
            [5.7f; 3.8f]; [5.1f; 3.8f]; [5.4f; 3.4f]; [5.1f; 3.7f]; [5.1f; 3.3f]; [4.8f; 3.4f];
            [5.0f; 3.0f]; [5.0f; 3.4f]; [5.2f; 3.5f]; [5.2f; 3.4f]; [4.7f; 3.2f]; [4.8f; 3.1f];
            [5.4f; 3.4f]; [5.2f; 4.1f]; [5.5f; 4.2f]; [4.9f; 3.1f]; [5.0f; 3.2f]; [5.5f; 3.5f];
            [4.9f; 3.6f]; [4.4f; 3.0f]; [5.1f; 3.4f]; [5.0f; 3.5f]; [4.5f; 2.3f]; [4.4f; 3.2f];
            [5.0f; 3.5f]; [5.1f; 3.8f]; [4.8f; 3.0f]; [5.1f; 3.8f]; [4.6f; 3.2f]; [5.3f; 3.7f];
            [5.0f; 3.3f]; [7.0f; 3.2f]; [6.4f; 3.2f]; [6.9f; 3.1f]; [5.5f; 2.3f]; [6.5f; 2.8f];
            [5.7f; 2.8f]; [6.3f; 3.3f]; [4.9f; 2.4f]; [6.6f; 2.9f]; [5.2f; 2.7f]; [5.0f; 2.0f];
            [5.9f; 3.0f]; [6.0f; 2.2f]; [6.1f; 2.9f]; [5.6f; 2.9f]; [6.7f; 3.1f]; [5.6f; 3.0f];
            [5.8f; 2.7f]; [6.2f; 2.2f]; [5.6f; 2.5f]; [5.9f; 3.0f]; [6.1f; 2.8f]; [6.3f; 2.5f];
            [6.1f; 2.8f]; [6.4f; 2.9f]; [6.6f; 3.0f]; [6.8f; 2.8f]; [6.7f; 3.0f]; [6.0f; 2.9f];
            [5.7f; 2.6f]; [5.5f; 2.4f]; [5.5f; 2.4f]; [5.8f; 2.7f]; [6.0f; 2.7f]; [5.4f; 3.0f];
            [6.0f; 3.4f]; [6.7f; 3.1f]; [6.3f; 2.3f]; [5.6f; 3.0f]; [5.5f; 2.5f]; [5.5f; 2.6f];
            [6.1f; 3.0f]; [5.8f; 2.6f]; [5.0f; 2.3f]; [5.6f; 2.7f]; [5.7f; 3.0f]; [5.7f; 2.9f];
            [6.2f; 2.9f]; [5.1f; 2.5f]; [5.7f; 2.8f]; [6.3f; 3.3f]; [5.8f; 2.7f]; [7.1f; 3.0f];
            [6.3f; 2.9f]; [6.5f; 3.0f]; [7.6f; 3.0f]; [4.9f; 2.5f]; [7.3f; 2.9f]; [6.7f; 2.5f];
            [7.2f; 3.6f]; [6.5f; 3.2f]; [6.4f; 2.7f]; [6.8f; 3.0f]; [5.7f; 2.5f]; [5.8f; 2.8f];
            [6.4f; 3.2f]; [6.5f; 3.0f]; [7.7f; 3.8f]; [7.7f; 2.6f]; [6.0f; 2.2f]; [6.9f; 3.2f];
            [5.6f; 2.8f]; [7.7f; 2.8f]; [6.3f; 2.7f]; [6.7f; 3.3f]; [7.2f; 3.2f]; [6.2f; 2.8f];
            [6.1f; 3.0f]; [6.4f; 2.8f]; [7.2f; 3.0f]; [7.4f; 2.8f]; [7.9f; 3.8f]; [6.4f; 2.8f];
            [6.3f; 2.8f]; [6.1f; 2.6f]; [7.7f; 3.0f]; [6.3f; 3.4f]; [6.4f; 3.1f]; [6.0f; 3.0f];
            [6.9f; 3.1f]; [6.7f; 3.1f]; [6.9f; 3.1f]; [5.8f; 2.7f]; [6.8f; 3.2f]; [6.7f; 3.3f];
            [6.7f; 3.0f]; [6.3f; 2.5f]; [6.5f; 3.0f]; [6.2f; 3.4f]; [5.9f; 3.0f]; [5.8f; 3.0f]])

        let y = np.array([|
            0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0;
            0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0;
            0; 0; 0; 0; 0; 0; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1;
            1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1;
            1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2;
            2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2;
            2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2 |])

        let url = "https://raw.githubusercontent.com/SciSharp/TensorFlow.NET/master/data/nb_example.npy";
        Web.Download(url, "nb", "nb_example.npy") |> ignore
        
        X, y

    let private fit (X : NDArray) (y : NDArray) =
        // Separate training points by class 
        // Shape : nb_classes * nb_samples * nb_features

        let dic : Map<int, float32 list list> =
            [ 0 .. y.size - 1 ]
            |> List.map (fun i -> int y.[i], [ float32 X.[i, 0]; float32 X.[i, 1] ])
            |> List.groupBy fst
            |> List.map (fun (curClass, values) -> curClass, values |> List.map snd)
            |> List.fold (fun map -> map.Add) Map.empty
        
        let maxCount =
            Map.toSeq dic
            |> Seq.map (snd >> List.length)
            |> Seq.max

        let points = Array3D.create dic.Count maxCount X.shape.[1] 0.f
        Map.toSeq dic |> Seq.iter (fun (j, values) ->
            values |> List.iteri (fun i pairs ->
                pairs |> List.iteri (fun k v ->
                    points.[j, i, k] <- v)))

        let points_by_class = np.array(points)
        // estimate mean and variance for each class / feature
        // shape : nb_classes * nb_features
        let cons = tf.constant(points_by_class)
        let struct (mean, variance) = tf.nn.moments(cons, [| 1 |])

        // Create a 3x2 univariate normal distribution with the 
        // Known mean and variance
        tf.distributions.Normal(mean, tf.sqrt(variance))

    let private predict (dist : Normal) (X : NDArray) =

        let nb_classes = int <| dist.scale().shape.[0]
        let nb_features = int <| dist.scale().shape.[1]

        // Conditional probabilities log P(x|c) with shape
        // (nb_samples, nb_classes)
        let t1 = ops.convert_to_tensor(X, TF_DataType.TF_FLOAT)
        let t2 = ops.convert_to_tensor([| 1; nb_classes |])
        let tile = tf.tile(t1, t2)
        let t3 = ops.convert_to_tensor([| -1; nb_classes; nb_features |])
        let r = tf.reshape(tile, t3)
        let cond_probs = tf.reduce_sum(dist.log_prob(r), 2)
        // uniform priors
        let tem = np.array<float32>(Array.create nb_classes (1.0f / float32 nb_classes))
        let priors = np.log(&tem)

        // posterior log probability, log P(c) + log P(x|c)
        let joint_likelihood = tf.add(ops.convert_to_tensor(priors, TF_DataType.TF_FLOAT), cond_probs)
        // normalize to get (log)-probabilities

        let norm_factor = tf.reduce_logsumexp(joint_likelihood, [| 1 |], keepdims = true)
        let log_prob = joint_likelihood - norm_factor
        // exp to get the actual probabilities
        tf.exp(log_prob)

    let private run () =
        tf.compat.v1.disable_eager_execution()
        let X, y = prepareData()

        let dist = fit X y

        // Create a regular grid and classify each point 
        //let x_min = X.amin(0).Data<float32>().[0] - 0.5f
        //let y_min = X.amin(0).Data<float32>().[1] - 0.5f
        //let x_max = X.amax(0).Data<float32>().[1] + 0.5f
        //let y_max = X.amax(0).Data<float32>().[1] + 0.5f

        //let struct (xx, yy) = np.meshgrid(np.linspace(x_min, x_max, 30), np.linspace(y_min, y_max, 30))
        //let samples = np.vstack<float32>(xx.ravel(), yy.ravel())
        //samples = np.transpose(samples)

        use sess = tf.Session()
        let array = np.Load<double[,]>(Path.Join("nb", "nb_example.npy"))
        let samples = np.array(array).astype(np.float32)
        //let Z = 
        sess.run(predict dist samples) |> ignore

        true

    let Example = { Config = ExampleConfig.Create("Naive Bayes Classifier", priority0 = 7)
                    Run = run }

