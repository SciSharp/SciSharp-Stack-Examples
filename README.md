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

* [Hello World](src/TensorFlowNET.Examples/HelloWorld.cs)
* [Basic Operations](src/TensorFlowNET.Examples/BasicOperations.cs)
* [Linear Regression](src/TensorFlowNET.Examples/BasicModels/LinearRegression.cs) in Graph mode
* [Linear Regression](src/TensorFlowNET.Examples/BasicModels/LinearRegressionEager.cs) in Eager mode
* [Logistic Regression](src/TensorFlowNET.Examples/BasicModels/LogisticRegression.cs) in Graph mode
* [Logistic Regression](src/TensorFlowNET.Examples/BasicModels/LogisticRegressionEager.cs) in Eager mode
* [Nearest Neighbor](src/TensorFlowNET.Examples/BasicModels/NearestNeighbor.cs)
* [Naive Bayes Classification](src/TensorFlowNET.Examples/BasicModels/NaiveBayesClassifier.cs)
* [Full Connected Neural Network](src/TensorFlowNET.Examples/\NeuralNetworks/FullyConnectedEager.cs) in Eager mode
* [K-means Clustering](src/TensorFlowNET.Examples/BasicModels/KMeansClustering.cs)
* [NN XOR](src/TensorFlowNET.Examples/NeuralNetworks/NeuralNetXor.cs)
* [Object Detection](src/TensorFlowNET.Examples/ObjectDetection/DetectInMobilenet.cs) in MobileNet
* [Binary Text Classification](src/TensorFlowNET.Examples/TextProcessing/BinaryTextClassification.cs)
* [CNN Text Classification](src/TensorFlowNET.Examples/TextProcessing/cnn_models/VdCnn.cs)
* [MNIST FNN](src/TensorFlowNET.Examples/ImageProcessing/MnistFnnKerasFunctional.cs) in Keras Functional API
* [MNIST CNN](src/TensorFlowNET.Examples/ImageProcessing/DigitRecognitionCNN.cs) in Graph mode
* [MNIST RNN](src/TensorFlowNET.Examples/ImageProcessing/DigitRecognitionRNN.cs)
* [MNIST LSTM](src/TensorFlowNET.Examples/ImageProcessing/DigitRecognitionLSTM.cs)
* [Named Entity Recognition](src/TensorFlowNET.Examples/TextProcessing/NER)
* [Transfer Learning for Image Classification](src/TensorFlowNET.Examples/ImageProcessing/TransferLearningWithInceptionV3.cs) in InceptionV3
* [CNN In Your Own Dataset](src/TensorFlowNET.Examples/ImageProcessing/CnnInYourOwnData.cs)


### Welcome to PR your example to us.
Your contribution will make .NET community better than ever.

TensorFlow.NET is a part of [SciSharp STACK](https://scisharp.github.io/SciSharp/)
<br>
<a href="http://scisharpstack.org"><img src="https://github.com/SciSharp/SciSharp/blob/master/art/scisharp-stack.png" width="391" height="100" /></a>
