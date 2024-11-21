# Neural Network Playground

## Introduction
This is a python desktop application that allows users to create and train an artificial neural network and recognize input handwritten digits using the network with a simple GUI. The neural network is trained solely on MNIST dataset, a well-known dataset of handwritten digits. The trained model can be saved and loaded as a .model file.

## Installation
To install the application, simply download the zip file from releases and extract it. Then open the app.exe in the extracted folder. The application has only Windows release for now.

## Usage
The application has a simple GUI with two main interfaces: the predict interface and the train interface.

1. **Predict Interface**:
    Users can write a digit in a writing board in this interface and click the "Predict" button to see the predicted digit using the loaded or trained model. The confidences of each number are also displayed. Note that the predicted result may not be correct for various reasons.

2. **Train Interface**:
    The train interface offers a convenient way to dynamically modify and train your neural network. Before training, click "Load Dataset" button to load the MNIST dataset. Then using the "Add Layer" and "Remove Layer" buttons to modify the structure of the neural network. The neuron counts and activation function of each hidden layer can also be modified. 

    After everything is set, click "Train" button to start the training process. The training progress and info will be displayed. After the training is done, the user can test the trained model in the predict interface. Note that when you tweaked the model params after training, the model's trained weights and biases will be reset.

## Notes
Due to the nature of the MNIST dataset, the trained model may not perform well on your writing input for various reasons like the varying size and position of the number and different writing styles. 

Since the application uses PyQt5 library, the size of the release may be a little large. If you have any methods to reduce the size of the application, please write an issue.

I'm still working on improving the application and adding more features. If you have any suggestions or feature requests, please write an issue.

## Credits
This artitecture of the neural network is based on the book [Neural Networks from Scratch in Python](https://nnfs.io/) by Harrison Kinsley & Daniel Kukieła. The interface is created using PyQt5 library and [QFluentWidgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets) library by zhiyiYo.