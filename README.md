# SciSharp STACK Examples
This repo contains many practical examples written in SciSharp's machine learning libraries. If you still don't know how to use .NET for deep learning, getting started from here is your best choice.

[![Join the chat at https://gitter.im/publiclab/publiclab](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/sci-sharp/community)



Requirements:

* [.NET Core 3.1](https://dotnet.microsoft.com/download/dotnet-core/3.1)

* [Visual Studio 2019](https://visualstudio.microsoft.com/vs/) or [Visual Studio Code](https://code.visualstudio.com/)



Run specific example in shell:

```cs
// run all examples from source code
dotnet run --project src/TensorFlowNET.Examples

// run specific example
dotnet run --project src/TensorFlowNET.Examples -ex "Linear Regression"

// run in compiled library
dotnet TensorFlowNET.Examples.dll -ex "MNIST CNN (Eager)"
```



Example runner will download all the required files like training data and model pb files.

* Hello World [C#](src/TensorFlowNET.Examples/HelloWorld.cs)
* Basic Operations [C#](src/TensorFlowNET.Examples/BasicOperations.cs)
* Linear Regression in Graph mode [C#](src/TensorFlowNET.Examples/BasicModels/LinearRegression.cs) 
* Linear Regression in Eager mode [C#](src/TensorFlowNET.Examples/BasicModels/LinearRegressionEager.cs), [F#](src/TensorFlowNET.Examples.FSharp/BasicModels/LinearRegressionEager.fs)
* Logistic Regression in Graph mode [C#](src/TensorFlowNET.Examples/BasicModels/LogisticRegression.cs)
* Logistic Regression in Eager mode [C#](src/TensorFlowNET.Examples/BasicModels/LogisticRegressionEager.cs) 
* Nearest Neighbor [C#](src/TensorFlowNET.Examples/BasicModels/NearestNeighbor.cs)
* Naive Bayes Classification [C#](src/TensorFlowNET.Examples/BasicModels/NaiveBayesClassifier.cs)
* Full Connected Neural Network in Eager mode [C#](src/TensorFlowNET.Examples/\NeuralNetworks/FullyConnectedEager.cs) 
* K-means Clustering [C#](src/TensorFlowNET.Examples/BasicModels/KMeansClustering.cs)
* NN XOR [C#](src/TensorFlowNET.Examples/NeuralNetworks/NeuralNetXor.cs)
* Object Detection in MobileNet [C#](src/TensorFlowNET.Examples/ObjectDetection/DetectInMobilenet.cs) 
* Binary Text Classification [C#](src/TensorFlowNET.Examples/TextProcessing/BinaryTextClassification.cs)
* CNN Text Classification [C#](src/TensorFlowNET.Examples/TextProcessing/cnn_models/VdCnn.cs)
* MNIST FNN in Keras Functional API [C#](src/TensorFlowNET.Examples/ImageProcessing/MnistFnnKerasFunctional.cs) 
* MNIST CNN in Graph mode [C#](src/TensorFlowNET.Examples/ImageProcessing/DigitRecognitionCNN.cs) 
* MNIST RNN [C#](src/TensorFlowNET.Examples/ImageProcessing/DigitRecognitionRNN.cs)
* MNIST LSTM [C#](src/TensorFlowNET.Examples/ImageProcessing/DigitRecognitionLSTM.cs)
* Named Entity Recognition [C#](src/TensorFlowNET.Examples/TextProcessing/NER)
* Transfer Learning for Image Classification in InceptionV3 [C#](src/TensorFlowNET.Examples/ImageProcessing/TransferLearningWithInceptionV3.cs)
* CNN In Your Own Dataset [C#](src/TensorFlowNET.Examples/ImageProcessing/CnnInYourOwnData.cs)


### Welcome to PR your example to us.
Your contribution will make .NET community better than ever.
<br>
<a href="http://scisharpstack.org"><img src="https://github.com/SciSharp/SciSharp/blob/master/art/scisharp-stack.png" width="391" height="100" /></a>