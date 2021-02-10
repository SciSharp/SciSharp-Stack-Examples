# SciSharp STACK Examples
This repo contains many practical examples written in SciSharp's machine learning libraries. If you still don't know how to use .NET for deep learning, getting started from these examples is your best choice.

[![Join the chat at https://gitter.im/publiclab/publiclab](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/sci-sharp/community)



Requirements:

* [.NET Core 3.1](https://dotnet.microsoft.com/download/dotnet-core/3.1)

* [Visual Studio 2019](https://visualstudio.microsoft.com/vs/) or [Visual Studio Code](https://code.visualstudio.com/)



Run specific example in shell:

#### C#

```bat
:: run all examples from source code
dotnet run --project src/TensorFlowNET.Examples

:: run specific example
dotnet run --project src/TensorFlowNET.Examples -ex "Linear Regression (Graph)"

:: run in compiled library
dotnet TensorFlowNET.Examples.dll -ex "MNIST CNN (Eager)"
```

#### F#

```bat
:: run all examples from source code
dotnet run --project src/TensorFlowNET.Examples.FSharp

:: run specific example
dotnet run --project src/TensorFlowNET.Examples.FSharp -ex "Linear Regression (Eager)"

:: run in compiled library
dotnet TensorFlowNET.Examples.FSharp.dll -ex "MNIST CNN (Eager)"
```


Example runner will download all the required files like training data and model pb files.

#### Basic Model

* Hello World [C#](src/TensorFlowNET.Examples/HelloWorld.cs), [F#](src/TensorFlowNET.Examples.FSharp/HelloWorld.fs)
* Basic Operations [C#](src/TensorFlowNET.Examples/BasicOperations.cs), [F#](src/TensorFlowNET.Examples.FSharp/BasicOperations.fs)
* Linear Regression in Graph mode [C#](src/TensorFlowNET.Examples/BasicModels/LinearRegression.cs), [F#](src/TensorFlowNET.Examples.FSharp/BasicModels/LinearRegression.fs)
* Linear Regression in Eager mode [C#](src/TensorFlowNET.Examples/BasicModels/LinearRegressionEager.cs), [F#](src/TensorFlowNET.Examples.FSharp/BasicModels/LinearRegressionEager.fs)
* Logistic Regression in Graph mode [C#](src/TensorFlowNET.Examples/BasicModels/LogisticRegression.cs), [F#](src/TensorFlowNET.Examples.FSharp/BasicModels/LogisticRegression.fs)
* Logistic Regression in Eager mode [C#](src/TensorFlowNET.Examples/BasicModels/LogisticRegressionEager.cs), [F#](src/TensorFlowNET.Examples.FSharp/BasicModels/LogisticRegressionEager.fs)
* Nearest Neighbor [C#](src/TensorFlowNET.Examples/BasicModels/NearestNeighbor.cs), [F#](src/TensorFlowNET.Examples.FSharp/BasicModels/NearestNeighbor.fs)
* Naive Bayes Classification [C#](src/TensorFlowNET.Examples/BasicModels/NaiveBayesClassifier.cs), [F#](src/TensorFlowNET.Examples.FSharp/BasicModels/NaiveBayesClassifier.fs)
* K-means Clustering [C#](src/TensorFlowNET.Examples/BasicModels/KMeansClustering.cs)

#### Neural Network

* Full Connected Neural Network in Eager mode [C#](src/TensorFlowNET.Examples/NeuralNetworks/FullyConnectedEager.cs), [F#](src/TensorFlowNET.Examples.FSharp/NeuralNetworks/FullyConnectedEager.fs)
* Full Connected Neural Network (Keras) [C#](src/TensorFlowNET.Examples/NeuralNetworks/FullyConnectedKeras.cs), [F#](src/TensorFlowNET.Examples.FSharp/NeuralNetworks/FullyConnectedKeras.fs)
* NN XOR [C#](src/TensorFlowNET.Examples/NeuralNetworks/NeuralNetXor.cs)
* Object Detection in MobileNet [C#](src/TensorFlowNET.Examples/ObjectDetection/DetectInMobilenet.cs) 
* MNIST FNN in Keras Functional API [C#](src/TensorFlowNET.Examples/ImageProcessing/MnistFnnKerasFunctional.cs) 
* MNIST CNN in Graph mode [C#](src/TensorFlowNET.Examples/ImageProcessing/DigitRecognitionCNN.cs) 
* MNIST RNN [C#](src/TensorFlowNET.Examples/ImageProcessing/DigitRecognitionRNN.cs)
* MNIST LSTM [C#](src/TensorFlowNET.Examples/ImageProcessing/DigitRecognitionLSTM.cs)
* Image Classification in Keras Sequential API [C#](src/TensorFlowNET.Examples/ImageProcessing/ImageClassificationKeras.cs)
* Toy ResNet in Keras Functional API [C#](src/TensorFlowNET.Examples/ImageProcessing/ToyResNet.cs)
* Transfer Learning for Image Classification in InceptionV3 [C#](src/TensorFlowNET.Examples/ImageProcessing/TransferLearningWithInceptionV3.cs)
* CNN In Your Own Dataset [C#](src/TensorFlowNET.Examples/ImageProcessing/CnnInYourOwnData.cs)

#### Natural Language Processing
* Binary Text Classification [C#](src/TensorFlowNET.Examples/TextProcessing/BinaryTextClassification.cs)
* CNN Text Classification [C#](src/TensorFlowNET.Examples/TextProcessing/cnn_models/VdCnn.cs)
* Named Entity Recognition [C#](src/TensorFlowNET.Examples/TextProcessing/NER)


#### Welcome to PR your example to us.
Your contribution will make .NET community better than ever.
<br>
<a href="http://scisharpstack.org"><img src="https://github.com/SciSharp/SciSharp/blob/master/art/scisharp-stack.png" width="391" height="100" /></a>