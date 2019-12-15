# TensorFlow.NET Examples
TensorFlow.NET Examples contains many practical examples written in C#. If you still don't know how to use .NET for deep learning, getting started from this Repo is your best choice.

[![Join the chat at https://gitter.im/publiclab/publiclab](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/sci-sharp/community)



Requirements:

* .NET Core 3.1

* Visual Studio 2019 (v16.4)

  

Run specific example in shell:

```cs
dotnet TensorFlowNET.Examples.dll -ex "MNIST CNN"
```



Example runner will download all the required files like training data and model pb files.

* [Hello World](src/TensorFlowNET.Examples/HelloWorld.cs)
* [Basic Operations](src/TensorFlowNET.Examples/BasicOperations.cs)
* [Linear Regression](src/TensorFlowNET.Examples/BasicModels/LinearRegression.cs)
* [Logistic Regression](src/TensorFlowNET.Examples/BasicModels/LogisticRegression.cs)
* [Nearest Neighbor](src/TensorFlowNET.Examples/BasicModels/NearestNeighbor.cs)
* [Naive Bayes Classification](src/TensorFlowNET.Examples/BasicModels/NaiveBayesClassifier.cs)
* [Full Connected Neural Network](src/TensorFlowNET.Examples/ImageProcess/DigitRecognitionNN.cs)
* [Image Processing](src/TensorFlowNET.Examples/ImageProcessing)
* [K-means Clustering](src/TensorFlowNET.Examples/BasicModels/KMeansClustering.cs)
* [NN XOR](src/TensorFlowNET.Examples/BasicModels/NeuralNetXor.cs)
* [Object Detection](src/TensorFlowNET.Examples/ImageProcessing/ObjectDetection.cs)
* [Text Classification](src/TensorFlowNET.Examples/TextProcessing/BinaryTextClassification.cs)
* [CNN Text Classification](src/TensorFlowNET.Examples/TextProcessing/cnn_models/VdCnn.cs)
* [MNIST CNN](src/TensorFlowNET.Examples/ImageProcessing/DigitRecognitionCNN.cs)
* [MNIST RNN](src/TensorFlowNET.Examples/ImageProcessing/DigitRecognitionRNN.cs)
* [MNIST LSTM](src/TensorFlowNET.Examples/ImageProcessing/DigitRecognitionLSTM.cs)
* [Named Entity Recognition](src/TensorFlowNET.Examples/TextProcessing/NER)
* [Transfer Learning for Image Classification in InceptionV3](src/TensorFlowNET.Examples/ImageProcessing/RetrainClassifierWithInceptionV3.cs)
* [Cnn In Your Own Data](src/TensorFlowNET.Examples/CnnInYourOwnData/CnnInYourOwnData.cs)



TensorFlow.NET is a part of [SciSharp STACK](https://scisharp.github.io/SciSharp/)
<br>
<a href="http://scisharpstack.org"><img src="https://github.com/SciSharp/SciSharp/blob/master/art/scisharp-stack.png" width="391" height="100" /></a>
